package main

import "fmt"

func main() {
	// Kode program fungsi z = Wx + b
    // Define the matrices and vectors
    var W [2][2]float64 = [2][2]float64{{1.2, 3.4}, {5.6, 7.8}}
    var x [2]float64 = [2]float64{9.1, 2.7}
    var b [2]float64 = [2]float64{1.2, 2.3}

    // Calculate Wx
    var wx [2]float64
    for i := 0; i < len(W); i++ {
        var sum float64 = 0
        for j := 0; j < len(W[i]); j++ {
            sum += W[i][j] * x[j]
        }
        wx[i] = sum
    }

    // Calculate z = Wx + b
    var z [2]float64
    for i := 0; i < len(wx); i++ {
        z[i] = wx[i] + b[i]
    }

    // Print the result
    fmt.Printf("z = [%.2f, %.2f]\n", z[0], z[1])

	// Penjelasan
	// Dalam kode ini, pertama-tama kita mendefinisikan matriks dan vektor W, x, dan b sebagai array. Kemudian, kita menghitung vektor Wx dengan mengalikan W dan x menggunakan loop bersarang. Akhirnya, kami menghitung z dengan menambahkan Wx dan b, dan mencetak hasilnya.
}
