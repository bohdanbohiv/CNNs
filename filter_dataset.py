import os

from PIL import Image


def main():
    data_dir = 'PetImages'
    for cat in ('Cat', 'Dog'):
        for file in os.listdir(os.path.join(data_dir, cat)):
            filename = os.path.join(data_dir, cat, file)
            try:
                Image.open(filename, formats=('JPEG',))
            except OSError:
                os.remove(filename)
                print(filename + ' is invalid, removing')


if __name__ == '__main__':
    main()
