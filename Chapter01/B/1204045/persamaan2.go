package main

import "fmt"

func matvec(W [][]float64, x []float64) []float64 {
	n := len(W)
    z := make([]float64, n)
	
    for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			z[i] += W[i][j] * x[j]
        }
    }
	
    return z
}

func main() {
	// Kode program fungsi z = Wx dimana W adalah matriks n kali n. x adalah vektor ukuran n. dengan n = 2.1957
    // Inisialisasi matriks W dan vektor x
    W := [][]float64{{1, 2}, {3, 4}}
    x := []float64{5, 6}

    // Panggil fungsi matvec untuk mengalikan W dengan x dan simpan hasilnya di variabel z
    z := matvec(W, x)

    // Print hasil perkalian
    fmt.Println("Hasil perkalian matriks dengan vektor:", z)

	// Persamaan
	// Di sini kita membuat fungsi matvec yang menerima dua parameter, yaitu sebuah matriks W dan sebuah vektor x. Fungsi ini mengembalikan hasil perkalian antara matriks W dengan vektor x dalam bentuk vektor baru z.

	// Di dalam fungsi matvec, kita menggunakan dua buah perulangan untuk mengalikan setiap elemen matriks W dengan elemen vektor x. Hasil perkalian disimpan dalam variabel z. Setelah selesai melakukan perulangan, kita mengembalikan variabel z.

	// Di dalam fungsi main, kita menginisialisasi matriks W dengan nilai 1 dan 2 untuk baris pertama dan nilai 3 dan 4 untuk baris kedua. Selanjutnya, kita menginisialisasi vektor x dengan nilai 5 dan 6. Kemudian, kita memanggil fungsi matvec dengan matriks W dan vektor x, dan menyimpan hasilnya dalam variabel z. Terakhir, kita mencetak hasil perkalian matriks dengan vektor dengan menggunakan fmt.Println().
}
