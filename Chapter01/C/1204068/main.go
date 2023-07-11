package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func main() {
	// Membuat sebuah graph baru
	g := gorgonia.NewGraph()

	// Membuat variabel x dan y sebagai input dari persamaan elips
	x := gorgonia.NodeFromAny(g, 3.0, gorgonia.WithName("x"))
	y := gorgonia.NodeFromAny(g, 2.0, gorgonia.WithName("y"))

	// Membuat konstanta a dan b sebagai parameter elips
	a := gorgonia.NodeFromAny(g, 2.0, gorgonia.WithName("a"))
	b := gorgonia.NodeFromAny(g, 1.0, gorgonia.WithName("b"))

	// Menghitung nilai persamaan elips
	xOverA := gorgonia.Must(gorgonia.Div(x, a))
	yOverB := gorgonia.Must(gorgonia.Div(y, b))
	squareX := gorgonia.Must(gorgonia.Square(xOverA))
	squareY := gorgonia.Must(gorgonia.Square(yOverB))
	sumSquares := gorgonia.Must(gorgonia.Add(squareX, squareY))
	result := gorgonia.Must(gorgonia.Sqrt(sumSquares))

	// Membuat output dari graph
	output := result

	// Membuat sebuah solver untuk menyelesaikan graph
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// Menyelesaikan graph dengan input x=3 dan y=2
	err := machine.RunAll()
	if err != nil {
		log.Fatal(err)
	}

	// Mengambil nilai output dari graph
	val := output.Value()
	if val == nil {
		log.Fatal("Failed to get value from graph")
	}

	// Menampilkan nilai persamaan elips
	fmt.Println(val)
}
