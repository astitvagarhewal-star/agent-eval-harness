package main

func SumChannel(nums []int) int {
	ch := make(chan int)
	go func() {
		var s int
		for _, n := range nums {
			s += n
		}
		ch <- s
	}()
	// BUG: never reads from the channel; the result is lost.
	return 0
}
