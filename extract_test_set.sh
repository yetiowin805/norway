# Combine whole.txt and split.txt, add random numbers, sort to shuffle
paste whole.txt split.txt | awk 'BEGIN {srand()} {print rand() "\t" $0}' | sort -n > temp.txt

# Calculate number of lines for the test set (10% of total lines)
total_lines=$(wc -l < whole.txt)
test_lines=$((total_lines / 10))

# Extract test set (first 10% of shuffled lines)
head -n $test_lines temp.txt | cut -f2 > test_whole.txt
head -n $test_lines temp.txt | cut -f3 > test_split.txt

# Extract residual (training) set (remaining 90% of shuffled lines)
tail -n +$((test_lines + 1)) temp.txt | cut -f2 > train_whole.txt
tail -n +$((test_lines + 1)) temp.txt | cut -f3 > train_split.txt

# Clean up temporary file
rm temp.txt