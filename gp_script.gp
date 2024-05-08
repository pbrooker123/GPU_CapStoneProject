# Extract arguments
plot_label = ARG1
x_title = ARG2
y_title = ARG3
file_name = ARG4
output_png = ARG5

set terminal png
set output output_png

# Set plot title and axis labels
set title plot_label
set xlabel x_title
set ylabel y_title

# Plot data from file_name
plot file_name using 1 with lines title 'Real Part', \
     '' using 2 with lines title 'Imaginary Part'
