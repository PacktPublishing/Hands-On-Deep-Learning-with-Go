package main

import "fmt"

func add(a int, b int) int {
    c := a + b
    return c
}

func main() {
    // Kode program fungsi : c = a + b
    // Panggil fungsi add dengan nilai 5 dan 10, simpan hasilnya di variabel sum
    sum := add(5, 10)

    // Print hasil penjumlahan
    fmt.Println("Hasil penjumlahan:", sum)

    // Penjelasan
    // Di sini kita membuat fungsi bernama add yang menerima dua parameter bertipe integer a dan b. Fungsi ini menambahkan kedua bilangan dan mengembalikan hasilnya dalam bentuk integer yang disimpan dalam variabel c.

    // Di dalam fungsi main, kita memanggil fungsi add dengan nilai 5 dan 10, dan menyimpan hasilnya dalam variabel sum. Kemudian, kita mencetak hasil penjumlahan dengan menggunakan fmt.Println().
}
