# Seam-Carving
A program for cropping images using dynamic programming to determine what pixels to cut from an image based the lowest values of pixels using their color gradient.

# Contribution / Method
I used OpenCV’s method for cropping the images to equal the size
of the pixels removed. I also used OpenCV to get the energy map
of the image using their Sobel operator. From the paper I used
dynamic programming M(i, j) = e(i, j)+ min(M(i−1, j −1),M(i−1,
j),M(i−1, j +1) to find the seams. This was the most difficult
part of the project, to compare the current pixel to the
previous pixel location, and then getting the full path of the
seam.

# Results
I tested a variety of images using 10 seams and 40 seams/pixels.
I just did 40 pixels for cropping since 10 was so minimal.


 
