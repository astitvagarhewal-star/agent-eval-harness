package main

func CopyInts(nums []int) []int {
	// BUG: aliases the input slice instead of copying it.
	return nums
}
