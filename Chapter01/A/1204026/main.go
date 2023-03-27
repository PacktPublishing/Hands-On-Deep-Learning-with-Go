package main

import (
	"fmt"
	"math"
)

func main() {
	// titik awal di garis
	x1 := 100.0
	y1 := 100.0

	// titik akhir di garis
	x2 := 300.0
	y2 := 200.0

	// hitung kemiringan garis
	theta := math.Atan((y2 - y1) / (x2 - x1))
	m := math.Tan(theta)

	// cetak persamaan garis dalam bentuk numerik
	fmt.Printf("y = %.2fx + %.2f\n", m, y1-m*x1)
}
