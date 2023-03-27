package main

import (
	"fmt"
	"time"
)

func main() {
	before := time.Now()
	max := 999
	for i := 100; i <= 999; i++ {
		for j := 100; j <= 999; j++ {
			product := i * j
			if isPalindrome(product) && product > max {
				max = product
			}
		}
	}
	fmt.Println(max)
	after := time.Now()
	fmt.Println("Waktu eksekusi ", (after.Nanosecond() - before.Nanosecond())/1_000_000, " millisecond")
}

func isPalindrome(n int) bool {
	reverse :=0
	originalNumber :=n
	for n !=0 {
		reverse =(reverse*10)+(n%10)
		n/=10
	}
	return originalNumber==reverse
}