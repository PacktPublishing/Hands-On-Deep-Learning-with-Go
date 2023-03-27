// package main

// import (
// 	"fmt"
// 	"time"
// )

// func main() {
// 	before := time.Now()
// 	sumSquares := 0
// 	squaresOfTheSum := 0
// 	numberOfData := 100
// 	for i := 1; i <= numberOfData; i++ {
// 		sumSquares += i * i
// 		squaresOfTheSum += i
// 	}
// 	diffSums := (squaresOfTheSum * squaresOfTheSum) - sumSquares
// 	fmt.Println("Selisis Jumlah Kuadrat Adalah : ", diffSums)
// 	after := time.Now()
// 	fmt.Println("Waktu eksekusi", after.Nanosecond()-before.Nanosecond(), "nano second")

// 	sumSquares = (numberOfData * (numberOfData + 1) * (2*numberOfData + 1)) / 6
// 	squaresOfTheSum = (numberOfData * (numberOfData + 1)) / 2
// }

package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func main() {
	// Persamaan garis elips: (x/a)^2 + (y/b)^2 = 1
	// Di sini, a dan b adalah parameter yang menentukan bentuk dan ukuran elips
	// Kita akan membuat sebuah tensor untuk menyimpan nilai a dan b
	params := gorgonia.NewTensor(gorgonia.Float64, 1, gorgonia.WithShape(2), gorgonia.WithName("params"))

	// Kita akan menginisialisasi nilai a dan b
	aVal := float64(2)
	bVal := float64(3)
	gorgonia.Let(params, []float64{aVal, bVal})

	// Kita akan membuat sebuah tensor untuk menyimpan nilai x dan y
	inputs := gorgonia.NewTensor(gorgonia.Float64, 1, gorgonia.WithShape(2), gorgonia.WithName("inputs"))

	// Kita akan menginisialisasi nilai x dan y
	xVal := float64(1)
	yVal := float64(2)
	gorgonia.Let(inputs, []float64{xVal, yVal})

	// Kita akan menghitung nilai z dari persamaan garis elips
	// Z = (x/a)^2 + (y/b)^2 - 1
	xOverA := gorgonia.Must(gorgonia.Div(inputs.At(0), params.At(0)))
	yOverB := gorgonia.Must(gorgonia.Div(inputs.At(1), params.At(1)))
	xOverASquared := gorgonia.Must(gorgonia.Square(xOverA))
	yOverBSquared := gorgonia.Must(gorgonia.Square(yOverB))
	one := gorgonia.NewScalar(float64(1))
	z := gorgonia.Must(gorgonia.Sub(gorgonia.Must(gorgonia.Add(xOverASquared, yOverBSquared)), one))

	// Kita akan membuat sebuah VM untuk mengeksekusi operasi yang telah kita buat
	vm := gorgonia.NewTapeMachine(gorgonia.WithWatchlist())

	// Kita akan mengeksekusi operasi yang telah kita buat
	if err := vm.RunAll(); err != nil {
		panic(err)
	}

	// Kita akan mencetak hasil dari operasi yang telah kita buat
	fmt.Println(z.Value())
}
