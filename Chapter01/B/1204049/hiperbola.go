package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func main() {
	// Membuat graph untuk perhitungan
	g := gorgonia.NewGraph()

	// Membuat variabel a dan b sebagai konstanta
	a := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))

	// Membuat variabel x dan y sebagai input
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))

	// Menghitung nilai z menggunakan persamaan garis hiperbola
	z1, err := gorgonia.Pow(x, gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithValue(2.0)))
	if err != nil {
		log.Fatal(err)
	}

	z1, err = gorgonia.Div(z1, a)
	if err != nil {
		log.Fatal(err)
	}
	z2, err := gorgonia.Pow(y, gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithValue(2.0)))
	if err != nil {
		log.Fatal(err)
	}
	z2, err = gorgonia.Div(z2, b)
	if err != nil {
		log.Fatal(err)
	}
	z2, err = gorgonia.Neg(z2)
	if err != nil {
		log.Fatal(err)
	}
	z, err := gorgonia.Add(z1, z2)
	if err != nil {
		log.Fatal(err)
	}
	z, err = gorgonia.Sqrt(z)
	if err != nil {
		log.Fatal(err)
	}

	// Membuat sebuah mesin baru
	m := gorgonia.NewTapeMachine(g)

	// Assign nilai a dan b
	gorgonia.Let(a, 2.0)
	gorgonia.Let(b, 3.0)

	// Assign nilai x dan y
	gorgonia.Let(x, 1.0)
	gorgonia.Let(y, 1.0)

	// Menjalankan mesin dan menghitung nilai z
	if err = m.RunAll(); err != nil {
		log.Fatal(err)
	}

	// Mencetak hasil
	fmt.Println("Nilai a:", a.Value())
	fmt.Println("Nilai b:", b.Value())
	fmt.Println("Nilai x:", x.Value())
	fmt.Println("Nilai y:", y.Value())
	fmt.Println("Persamaan Hiperbola : x^2/a^2 - y^2/b^2 = 1")
	fmt.Println("Sehingga untuk mencari nilai z : z = √((x^2/a) - (y^2/b))")
	fmt.Println("Hasil z = ", z.Value())

}
