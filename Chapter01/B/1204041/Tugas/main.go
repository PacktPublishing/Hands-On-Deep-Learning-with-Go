package main

import (
	"fmt"

	_ "go4.org/unsafe/assume-no-moving-gc"

	"gorgonia.org/gorgonia"
)

func main() {
	// Membuat Graph baru
	g := gorgonia.NewGraph()

	// Membuat input node untuk variabel x dan y
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))

	// Membuat operasi pembagian pada Graph
	div := gorgonia.Must(gorgonia.Div(x, y))

	// Membuat pembentuk Graph
	machine := gorgonia.NewTapeMachine(g)

	// Mengisi nilai input node x dan y
	xVal := 10.0
	yVal := 5.0

	// Bind nilai x dan y ke dalam node input
	gorgonia.Let(x, xVal)
	gorgonia.Let(y, yVal)

	// Menjalankan pembentuk Graph
	if err := machine.RunAll(); err != nil {
		fmt.Println(err)
	}

	// Mengambil nilai hasil pembagian dari node output
	result := div.Value().Data().(float64)

	// Mencetak hasil pembagian
	fmt.Printf("%v/%v = %v\n", xVal, yVal, result)
}
