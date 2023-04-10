package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	// Membuat graph baru
	g := gorgonia.NewGraph()

	// Membuat node input x
	x := gorgonia.NewTensor(g, tensor.Float64, 1, gorgonia.WithName("x"), gorgonia.WithShape(4), gorgonia.WithInit(gorgonia.Gaussian(0, 1)))

	// Menghitung nilai sin(x)
	sinX, err := gorgonia.Sin(x)
	if err != nil {
		log.Fatal(err)
	}

	// Membuat VM baru
	machine := gorgonia.NewTapeMachine(g)

	// Menjalankan VM
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	// Mengambil hasil
	result := sinX.Value().Data().([]float64)

	// Mencetak hasil
	fmt.Println(result)

	// Penjelasan singkat tentang program di atas:

		// Pertama, program membuat graph baru menggunakan gorgonia.NewGraph().
		// Selanjutnya, program membuat node input x menggunakan gorgonia.NewTensor() dengan bentuk [4].
		// Program kemudian menghitung nilai sin(x) menggunakan gorgonia.Sin().
		// Selanjutnya, program membuat VM baru menggunakan gorgonia.NewTapeMachine().
		// Program menjalankan VM menggunakan machine.RunAll().
		// Akhirnya, program mengambil hasil menggunakan sinX.Value().Data() dan mencetak hasil tersebut menggunakan fmt.Println().
		// Dalam contoh program di atas, kita menghitung nilai sin(x) di mana x adalah tensor dengan bentuk [4] yang diinisialisasi dengan distribusi Gaussian. Hasilnya adalah tensor [sin(x[0]), sin(x[1]), sin(x[2]), sin(x[3])].
}
