import os


INPUT_DIRECTORY = 'input_data'

DISTRIBUTION_FILES = [
    'bin_test.bin',
    'cells.jpg',
    'csv_test.csv',
    'json_test.json',
    'txt_test.txt',
]

PATH_TO_IMG = os.path.join(INPUT_DIRECTORY, 'sar_1_gray.jpg')

PYTHON_PATH = rf'D:\miniconda3\envs\tf_env\python'

OUTPUT_DIRECTORY = 'output_data'

for file in DISTRIBUTION_FILES:
    os.system(
        rf'{PYTHON_PATH} '
        rf'.\test_readers.py -img {PATH_TO_IMG} '
        rf'-p {os.path.join(INPUT_DIRECTORY, file)} '
        rf'-o {os.path.join(OUTPUT_DIRECTORY, file) + ".jpg"}'
    )

gamma_corr_path = os.path.join(OUTPUT_DIRECTORY, "gamma-correction")

for alpha in range(5, 16, 5):
    alpha /= 10
    for beta in range(-50, 60, 50):
        os.system(
            rf'{PYTHON_PATH} '
            rf'.\test_readers.py -img {PATH_TO_IMG} '
            rf'-c gamma-correction -a {alpha} -b {beta} '
            rf'-o {gamma_corr_path}--alpha={alpha}--beta={beta}.jpg'
        )

os.system(
    rf'{PYTHON_PATH} '
    rf'.\test_readers.py -img {PATH_TO_IMG} '
    rf'-c equalization '
    rf'-o {os.path.join(OUTPUT_DIRECTORY, "equalized.jpg")}'
)