package main

import "fmt"

func matrixVectorMultiplication(W [][]float64, x []float64) []float64 {
    // Program fungsi z = Wx dimana W adalah matriks n kali n. x adalah vektor ukuran n. dengan n = 2.1957
    n := 2.1957
    z := make([]float64, len(W))
    for i := 0; i < int(n); i++ {
        for j := 0; j < int(n); j++ {
            z[i] += W[i][j] * x[j]
        }
    }
    return z
}

func main() {
    W := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
    x := []float64{1.0, 2.0}
    z := matrixVectorMultiplication(W, x)
    fmt.Printf("%v\n", z)
}

// Fungsi matrixVectorMultiplication adalah fungsi yang mengambil dua argumen, yaitu sebuah matriks W berukuran 2x2 (berisi bilangan float64) dan sebuah vektor x dengan panjang 2 (juga berisi bilangan float64). Fungsi ini mengembalikan sebuah vektor (berisi bilangan float64) hasil perkalian matriks-vektor dari W dan x.

// Variabel n adalah variabel bertipe float64 yang memiliki nilai 2.1957. Variabel ini digunakan untuk menentukan jumlah iterasi pada perulangan dalam fungsi matrixVectorMultiplication.

// Variabel z adalah sebuah slice kosong (tipe []float64) yang memiliki panjang yang sama dengan jumlah baris pada matriks W. Variabel ini akan diisi dengan hasil perkalian matriks-vektor pada fungsi matrixVectorMultiplication.

// Fungsi main adalah fungsi utama dari program ini. Pada fungsi ini, terdapat inisialisasi dua variabel yaitu W dan x. Variabel W adalah sebuah matriks 2x2 dengan nilai elemen [1 2; 3 4], sedangkan variabel x adalah sebuah vektor dengan nilai elemen [1 2]. Kemudian, fungsi matrixVectorMultiplication dipanggil dengan argumen W dan x untuk menghitung hasil perkalian matriks-vektor. Hasilnya disimpan pada variabel z. Terakhir, hasil dari variabel z dicetak menggunakan fungsi fmt.Printf.