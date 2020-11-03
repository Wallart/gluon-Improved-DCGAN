from tqdm import tqdm
from glob import glob

import os
import imageio
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build animated GIF from images')
    parser.add_argument('images_dir', type=str, help='Images directory')
    parser.add_argument('output_gif', type=str, help='Output GIF image')
    parser.add_argument('--ext', type=str, default='png', help='Images extension')
    parser.add_argument('--interval', type=int, default=5, help='Images extension')

    args = parser.parse_args()

    args.images_dir = os.path.expanduser(args.images_dir)
    args.output_gif = os.path.expanduser(args.output_gif)

    images = glob(os.path.join(args.images_dir, f'*.{args.ext}'))
    assert len(images) > 1, 'No images found.'

    idx = args.interval
    with imageio.get_writer(args.output_gif, mode='I') as writer:
        for _ in tqdm(range(len(images))):
            img_path = os.path.join(args.images_dir, f'epoch-{idx}.{args.ext}')
            idx += args.interval
            if not os.path.isfile(img_path):
                continue

            image = imageio.imread(img_path)
            writer.append_data(image)

    print(f'{args.output_gif} saved.')
