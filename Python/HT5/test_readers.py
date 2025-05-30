import sys
import argparse

from utils.reader import image_reader as imread
from utils.reader import csv_reader, bin_reader, txt_reader, json_reader
from utils.processor import histogram
from utils.writer import csv_writer, bin_writer, txt_writer, image_writer, json_writer

from utils.image_toner import stat_correction, equalization, gamma_correction


def print_args_1():
    print(type(sys.argv))
    if (len(sys.argv) > 1):
        for param in sys.argv[1:]:
            print(param, type(param))
    return sys.argv[1:]


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-img', '--img-path', required=True,
                        help='Path to input image', dest='img_path')

    parser.add_argument('-p', '--path', help='Distribution file path')
    parser.add_argument('-c', '--cmd',
                        help='Choose image conversion operation',
                        choices=['gamma-correction', 'equalization'])

    parser.add_argument('-a', '--alpha',
                        help='Alpha parameter in gamma-correction', type=float)
    parser.add_argument('-b', '--beta',
                        help='Beta parameter in gamma-correction', type=float)

    parser.add_argument('-o', '--output', required=True,
                        help='Output file path')

    return parser


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args(sys.argv[1:])

    DATA_PROCESSORS = {
        'csv': csv_reader,
        'bin': bin_reader,
        'txt': txt_reader,
        'json': json_reader,
    }

    if (args.path is None) == (args.cmd is None):
        raise ValueError(
            'None of or both "--path" & "--cmd" are specified. Set one of them')

    main_image = imread.read_data(args.img_path)

    res = distribution = None
    match (args.path or '').split('.')[-1]:
        case ext if ext in DATA_PROCESSORS:
            distribution = DATA_PROCESSORS[ext].read_data(args.path)
        case 'jpg' | 'jpeg' | 'png' | 'bmp':
            img = imread.read_data(args.path)
            distribution = histogram.image_processing(img)
        case _:
            pass

    if distribution:
        res = stat_correction.processing(distribution, main_image)

    match args.cmd:
        case 'equalization':
            res = equalization.equalize_image(main_image)
        case 'gamma-correction':
            if args.alpha is None or args.beta is None:
                raise ValueError(
                    'Both "--alpha" & "--beta" must be specified alongside "gamma-correction"')
            res = gamma_correction.apply_gamma_correction(
                main_image, args.alpha, args.beta)
        case _:
            pass

    match args.output.split('.')[-1]:
        case 'jpg' | 'jpeg' | 'png' | 'bmp':
            image_writer.write_data(args.output, res)
        case _:
            raise ValueError("Unknown format of output file")
