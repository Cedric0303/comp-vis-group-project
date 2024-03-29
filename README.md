# Files and folders:

- `project.ipynb` - main program code
- `Dataset` - input image pair dataset
- `images` - program output images and plots
- `statsval_25_50.png` - statistic values used in report table

# Statistics and program outputs:

1. All images and plots used in the report are in `images` folder.
2. Each folder represents images and plots created by the program, with format `BLOCKSIZE_SEARCHSIZE` in the folder name.
3. For example, `25_50` folder contains images and plots created by using pixel block size of 25 and sliding window search size of 50.
4. RMS value and pixel error fractions statistics values used in the report is `statsval_25_50.png`

# Instructions to run program:

1. Run each cell in `project.ipynb` sequentially to start the program.
2. Output images and statistic plots will be in `images` folder.
3. Currently the program is set to output of pixel block size of 25 and sliding window search size of 50.
4. (optional) Change values of `BLOCK_SIZE` and `SEARCH_SIZE` to create output disparity maps of different pixel block size and sliding window search size.
