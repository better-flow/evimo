#!/usr/bin/python3
import os, subprocess, argparse
from tqdm import tqdm

def compress(base_folder, fin, fout, fname):
    os.makedirs(fout, exist_ok=True)

    # Parallel compression with pigz
    compress_process = subprocess.Popen(
      ['tar', '-c', '--use-compress-program=pigz',
       '-f', os.path.join(fout, fname),
       '-C', base_folder,
       fin]
    )

    compress_process.wait()

def compress_all_list(idir, compressed_dir):
    os.makedirs(compressed_dir, exist_ok=True)

    base_list = []
    fin_list = []
    fout_list = []
    fname_list = []

    for g in os.listdir(os.path.join(idir)):
        for subgroup in os.listdir(os.path.join(idir, g)):
            for dtype in os.listdir(os.path.join(idir, g, subgroup)):
                if (g == 'raw'):
                    base_list.append(os.path.join(idir, g))
                    fin_list.append(os.path.join(subgroup, dtype))
                    fout_list.append(os.path.join(compressed_dir, g))
                    fname_list.append(g + '_' + dtype + '_' + c + '.tar.gz')
                else:
                    for c in os.listdir(os.path.join(idir, g, subgroup, dtype)):
                        base_list.append(os.path.join(idir, g))
                        fin_list.append(os.path.join(subgroup, dtype, c))
                        fout_list.append(os.path.join(compressed_dir, g))
                        fname_list.append(g + '_' + subgroup + '_' + dtype + '_' + c + '.tar.gz')

    return base_list, fin_list, fout_list, fname_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idir',
                        type=str)
    parser.add_argument('odir',
                        type=str)
    parser.add_argument('dry_compress',
                        type=str)
    args = parser.parse_args()

    if args.dry_compress == 'dry':
        dry_run = True
    elif args.dry_compress == 'compress':
        dry_run = False
    else:
        raise ValueError('Unknown value set for option dry_compress')

    base_list, fin_list, fout_list, fname_list = compress_all_list(args.idir, args.odir)

    print ("File list: ")
    for i, _ in enumerate(fin_list):
        print (i, "\t", base_list[i], "\t", fin_list[i], '->\t', fout_list[i], "\t", fname_list[i])

    print ("Compressing:")
    for i, _ in tqdm(enumerate(fin_list), total=len(fin_list), dynamic_ncols=True, maxinterval=0.001):
        if not dry_run:
            compress(base_list[i], fin_list[i], fout_list[i], fname_list[i])
