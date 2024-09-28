import argparse, sys, shutil, time
import glob, os, math, functools
import numpy as np
import einops
from pathlib import Path
from PIL import Image
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/data_compression/data_compression')
import torch
import torch.nn.functional as F
import torchac
import cv2
import struct
from data_compression.metric import ms_ssim
import our_codecs
from thop import profile
torch.backends.cudnn.deterministic = True

models = {
    'ours_groupswin_channelar': our_codecs.GroupChARTTC,
    'ours_groupswin_channelar_woStructure': our_codecs.GroupChARTTC_woStructure,
}

def load_img(p, padding=True, factor=64):
    x = Image.open(p)
    x = torch.from_numpy(np.asarray(x))
    x = x.permute(2, 0, 1).unsqueeze(0).float().div(255)
    h, w = x.shape[2:4]

    if padding:
        dh = factor * math.ceil(h / factor) - h
        dw = factor * math.ceil(w / factor) - w
        x = F.pad(x, (0, dw, 0, dh))
    return x, h, w

def pack_string(string):
    bit_stream = struct.pack(f'>I', len(string))
    bit_stream += struct.pack(f'>{len(string)}s', string)
    return bit_stream

def unpack_string(bit_stream):
    s1 = struct.calcsize('I')
    s2 = struct.calcsize('s')
    length = struct.unpack(f'>I', bit_stream[:s1])[0]
    string = struct.unpack(f'>{length}s', bit_stream[s1:s1+s2*length])[0]
    return string, bit_stream[s1+s2*length:]

def pack_strings(strings):
    bit_stream = b''
    for string in strings:
        bit_stream += pack_string(string)
    return bit_stream

def unpack_strings(bit_stream, n):
    strings = []
    for i in range(n):
        string, bit_stream = unpack_string(bit_stream)
        strings.append(string)
    return strings, bit_stream

def pack_uints(uints):
    bit_stream = struct.pack(f'>{len(uints)}I', *uints)
    return bit_stream

def unpack_uints(bit_stream, n):
    s1 = struct.calcsize('I')
    uints = struct.unpack(f'>{n}I', bit_stream[:n*s1])
    return uints, bit_stream[n*s1:]

def load_codec_state_dict(model, state_dict):
    model.load_state_dict(state_dict, strict=False)
    model.init_tables()
    model.load_state_dict(state_dict, strict=True)
    return model

def get_cdf_group_mask(h, w, ss_block_size, max_group_numbers=64):
    # --- use Normal distribution.
    dist_normal = torch.distributions.Normal(0, 16)
    cdf_group_msk = dist_normal.cdf(torch.arange(1, max_group_numbers + 1))
    cdf_group_msk = (cdf_group_msk - .5) * 2
    cdf_group_msk = einops.repeat(cdf_group_msk, 'Lp -> b c h w Lp',
                                  b=1, c=1, h=int(h / ss_block_size), w=int(w / ss_block_size))

    cdf_group_msk = F.pad(cdf_group_msk, (1, 0))
    return cdf_group_msk

def build_model(args):
    if 'ours' in args.model:
        if 'swin' in args.model:
            if args.swin_disable_norm:
                norm_layer = nn.Identity
            else:
                norm_layer = nn.LayerNorm
            model = models[args.model](norm_layer=norm_layer)
        elif 'group' in args.model:
            model = models[args.model](args.hyper_channels)
        else:
            model = models[args.model](args.transform_channels, args.hyper_channels)
    else:
        model = models[args.model](args.N, args.M)
    model.eval()
    model.cuda()
    return model

def compress(args):
    # Load model: init model -> load cdf shape -> init cdf tables
    # -> load cdf tables and all.
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.resume, map_location=args.device)
    try:
        model = build_model(args)
        load_codec_state_dict(model, ckpt['parameters'])
        model.to(args.device)
    except:
        model = build_model(args)
        model.load_state_dict(ckpt['parameters'], strict=False)
        with torch.set_grad_enabled(True):
            model.fix_tables()
        torch.save({
            'iteration': ckpt['iteration'],
            'configs': ckpt['configs'],
            'parameters': model.state_dict()
        },str(args.resume))
        print('Reinitialize cdf tables.')

    input_ps = sorted(glob.glob(args.input_file_glob.strip("'")))
    bpp, bpp_es, bpp_es_z, bpp_z, psnr, msssim, enc_time, dec_time, t_a, YDec, psnr_trans, ZDec= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    os.system('mkdir -p {}'.format(args.output_file_dir))

    for input_p in input_ps:
        name = Path(input_p).stem
        output_p = Path(args.output_file_dir) / name

        # Load image
        x, hx, wx = load_img(input_p, padding=True)
        x = x.to(args.device)
        b, c, h, w = x.shape

        ##################################### Encoding ##########################################
        if args.groupvit_load_group_mask: # load predefined group mask, you can generate it with your own way.
            msk = Image.open(os.path.join(args.groupvit_load_group_mask,
                             input_p.split('/')[-1])).convert('L')
            msk = np.asarray(msk).astype(np.uint8)
            msk[msk != 0] = 1
            _, msk, _, _ = cv2.connectedComponentsWithStats(
                msk, connectivity=4)
            group_mask = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).to(args.device)
        else:
            group_mask = torch.ones(1, 1, h//16, w//16).to(args.device)
        torch.cuda.synchronize()
        t0 = time.time()
        strings, enc_var = model.compress(x, group_mask)
        # Write a binary file.
        sideInfo = pack_uints((hx, wx))
        sideInfo += pack_string(strings.pop())
        with open(str(output_p) + '_sideInfo', "wb") as f:
            f.write(sideInfo)
        if 'woStructure' in args.model:
            bitstream = pack_strings(strings[0])
            with open(str(output_p) + '_woStructure', "wb") as f:
                f.write(bitstream)
        else:
            for i in group_mask.unique().tolist():
                bitstream = pack_strings(strings[0][i])
                with open(str(output_p) + f'_group_{i}', "wb") as f:
                    f.write(bitstream)
            # compress group mask
            ss_block_size = args.ss_block_size
            assert ss_block_size in [16, 32, 64, 128]
            bs_ss_block_size = pack_uints((ss_block_size,))
            GroupmaskInfo = bs_ss_block_size
            
            group_mask_scale = ss_block_size // 16
            group_mask = torch.nn.functional.interpolate(
                group_mask.float(), scale_factor=1/group_mask_scale, mode='nearest')
            assert len(group_mask.unique()) < args.max_group_numbers
            cdf_group_mask = get_cdf_group_mask(h, w, ss_block_size)
            sym = group_mask.short() + 1         # -1 means not compress.
            bs_group_mask = torchac.encode_float_cdf(cdf_group_mask, sym.to('cpu'), check_input_bounds=True)
            group_mask_length = len(bs_group_mask)
            bs_group_mask_length = pack_uints((group_mask_length, ))
            GroupmaskInfo += bs_group_mask_length
            GroupmaskInfo += bs_group_mask

            with open(str(output_p) + '_GroupmaskInfo', "wb") as f:
                f.write(GroupmaskInfo)
        torch.cuda.synchronize()
        enc_time += time.time() - t0

        ##################################### Decoding ##########################################
        torch.cuda.synchronize()
        t0 = time.time()
        bits_trans = 0
        with open(str(output_p) + '_sideInfo', "rb") as f:
            bit_stream_dec = f.read()
            bits_trans += len(bit_stream_dec)
        shape_dec, bit_stream_dec = unpack_uints(bit_stream_dec, 2)
        side_string_dec, bit_stream_dec = unpack_string(bit_stream_dec)

        if 'woStructure' in args.model:
            num_strings = 10 # 10 slices
            with open(str(output_p) + '_woStructure', "rb") as f:
                string_dec = f.read()
                bits_trans += len(string_dec)
            string_dec, _ = unpack_strings(string_dec, num_strings)
            print(f'shape_dec: {shape_dec}')
            group_mask = torch.ones(1, 1, shape_dec[0]//16, shape_dec[1]//16).to(args.device)
            x_hat, dec_var = model.decompress((string_dec, side_string_dec), shape_dec, group_mask=group_mask)
        else:
            # Load group mask
            with open(str(output_p) + '_GroupmaskInfo', "rb") as f:
                GroupmaskInfo = f.read()
                bits_trans += len(GroupmaskInfo)
            ss_block_size, GroupmaskInfo = unpack_uints(GroupmaskInfo, 1)
            ss_block_size = ss_block_size[0]
            group_mask_length, GroupmaskInfo = unpack_uints(GroupmaskInfo, 1)
            group_mask_length = group_mask_length[0]
            bs_group_mask = GroupmaskInfo[:group_mask_length]
            hx, wx = shape_dec
            padding_factor = 16
            h = int(np.ceil(hx/ padding_factor) * padding_factor)
            w = int(np.ceil(wx/ padding_factor) * padding_factor)
            cdf_group_mask = get_cdf_group_mask(h, w, ss_block_size)
            group_mask = torchac.decode_float_cdf(cdf_group_mask, bs_group_mask) - 1
            group_mask = torch.nn.functional.interpolate(
                group_mask.float(), scale_factor=ss_block_size//padding_factor, mode='nearest').short()
            group_mask = group_mask.to(args.device)

            num_strings = 10 # 10 slices
            if args.groups_tobe_decode == [0, 0, 0]:
                group_idxs = group_mask.unique().tolist()
                group_idxs = group_idxs[1:]
            elif args.groups_tobe_decode is not None:
                for group_idx in args.groups_tobe_decode:
                    assert group_idx in group_mask.unique()
                group_idxs = sorted(args.groups_tobe_decode)
            else:
                group_idxs = group_mask.unique().tolist()
                if -1 in group_idxs:
                    group_idxs.remove(-1)
            group_strings_dec =[]
            for i in group_idxs:
                with open(str(output_p) + f'_group_{i}', "rb") as f:
                    string_dec = f.read()
                    bits_trans += len(string_dec)
                group_string_dec, _ = unpack_strings(string_dec, num_strings)
                group_strings_dec.append(group_string_dec)
            x_hat, _ = model.decompress((group_strings_dec, side_string_dec), shape_dec, group_mask, group_idxs)

        torch.cuda.synchronize()
        dec_time += time.time() - t0
        num_pixels = hx * wx
        bpp += bits_trans * 8. / num_pixels

        x = x[:, :, :hx, :wx].clamp(0, 1).mul(255).round()
        x_hat = x_hat[:, :, :hx, :wx].clamp(0, 1).mul(255).round()

        ## Save image
        x_rec = x_hat.squeeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        x_rec = Image.fromarray(x_rec)
        if os.path.exists(f'{args.output_file_dir}/recon') is False:
            os.system('mkdir -p {}'.format(f'{args.output_file_dir}/recon'))
        x_rec.save(f'{args.output_file_dir}/recon/{name}.png')
        mse = ((x - x_hat)**2).mean().item()
        psnr += 20 * np.log10(255.) - 10 * np.log10(mse)
        msssim += ms_ssim(x, x_hat, data_range=255).item()
        # decoded region psnr
        psnr_trans = 0
        if 'woStructure' not in args.model:
            msk_trans = torch.ones_like(group_mask)
            msk_trans[~torch.isin(group_mask, torch.tensor(group_idxs).to(x.device))] = 0
            msk_trans = F.interpolate(msk_trans.float(), size=(hx, wx), mode='nearest').cuda()  # [1,1,hx,wx]
            psnr_trans += 20 * np.log10(255.) - 10 * torch.log10((((x - x_hat) ** 2) * msk_trans).sum() / (msk_trans.sum() * 3))

    if args.verbose:
        print(f"Number of images: {len(input_ps):0.0f}")
        print(f"mse: {mse/len(input_ps):0.2f}")
        print(f"Entire Image PSNR (dB): {psnr/len(input_ps):0.2f}")
        print(f"Partial Recon PSNR (dB): {psnr_trans/len(input_ps):0.2f}")
        print(f"Multiscale SSIM: {msssim/len(input_ps):0.4f}")
        print(f"Bits per pixel: {(bpp)/len(input_ps):0.4f}")
        print(f"Average encoding time (ms): {1000*enc_time/len(input_ps):0.0f}")
        print(f"Average decoding time (ms): {1000*dec_time/len(input_ps):0.0f}")

        input = torch.randn(1, 3, hx, wx).to(x.device)
        msk = torch.randn(1, 1, hx, wx).to(x.device)
        flops, params = profile(model, inputs=(input,msk))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')


def main(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "command", choices=["compress", "decompress"])
    args = parser.parse_args(argv[:1])

    if args.command == "compress":
        parser = argparse.ArgumentParser(description="Compress image to bit-stream")
        parser.add_argument(
            "--model", type=str, default="", choices=models.keys(),
            help="Model")
        parser.add_argument(
            '--resume', type=str, default='',
            help='Checkpoint path')
        parser.add_argument(
            "--groupvit_load_group_mask", type=str, default="",
            help="Group mask image path")
        parser.add_argument(
            "--input_file_glob", type=str, default="",
            help="Input image path")
        parser.add_argument(
            "--output_file_dir", type=str, default="",
            help="Output binary file")
        parser.add_argument(
            "--cp", type=float, default=0,
            help="Complexity parameter")
        parser.add_argument(
            '--verbose', action='store_true', default=False,
            help='Verbose')
        parser.add_argument(
            '-c', type=str, choices=['ans', 'rangeCoder'], default='ans',
            help='Entropy coder (default: ans)')
        parser.add_argument('--N', type=int, default=128)
        parser.add_argument('--M', type=int, default=192)
        parser.add_argument('--char_layernum', type=int, default=3)
        parser.add_argument('--char_embed', type=int, default=128)
        parser.add_argument("--transform-channels", type=int, nargs='+',
                            default=[128, 128, 128, 192],
                            help="Transform channels.")
        parser.add_argument("--hyper-channels", type=int, nargs='+',
                            default=None, help="Transform channels.")
        parser.add_argument("--depths_char", type=int, nargs='+',
                            default=[1, 1, 1],
                            help="Depth of GroupSwinBlocks in ChARM.")
        parser.add_argument("--num_heads_char", type=int, nargs='+',
                            default=[8, 8, 8],
                            help="Head num of GroupSwinBlocks in ChARM.")
        parser.add_argument('--swin-disable-norm', action='store_true',
                            help='do not use any normalization in the transformation.')
        parser.add_argument("--groups_tobe_decode", type=int, nargs='+',
                            default=[0,0,0], help="group idxs to be decoded.")
        parser.add_argument('--ss_block_size', type=int, default=32)
        parser.add_argument('--max_group_numbers', type=int, default=64)
        args = parser.parse_args(argv[1:])
        with torch.no_grad():
            compress(args)

if __name__ == "__main__":
    main(sys.argv[1:])
