# Algeo02-20010

# Tugas Besar Aljabar Linier dan Geometri Aplikasi Nilai Eigen dan Vektor Eigen dalam Kompresi Gambar (*Image Compression*)

## 13520010 - Ken Kalang Al Qalyubi
## 13520110 - Farrel Ahmad
## 13520143 - Muhammad Gerald Akbar Giffera 

<br>

## Table of Contents
- [Introduction](#intro)
- [Program Setup](#setup)
- [Program Example](#pe)

<br>

## Introduction <a name = "intro"></a>
Program ini adalah program kompresi gambar/*Image Compression* menggunakan metode SVD. SVD adalah *Singular Value Decomposition* sebuah metode dekomposisi matriks menjadi matriks U, Sigma, dan Vt. Dengan mengambil K kolom dari U, K diagonal Sigma, dan K baris dari Vt. Gambar tersebut direkonstruksi kembali melalui perkalinan matriks U * Sigma * Vt. Setelah itu, gambar dapat terkompresi karena semua informasi penting berada pada K awal-awal dari matriks-matriks dekomposisi tersebut.

Struktur directory :
- src : source code frontend dan backend
- doc : laporan
- test : gambar-gambar masukan dan beberapa hasil kompresi

<br>

## Program Setup <a name = "setup"></a>
1. Instalasi library dengan pip
```sh
$ python -m pip install numpy
$ python -m pip install sympy
$ python -m pip install opencv-python
$ python -m pip install flask
```
2. Clone repository ini.
```sh
$ git clone https://github.com/farrel-a/Algeo02-20010.git
```

3. Buka terminal dan jalankan command ini.
```sh
$ cd Algeo02-20010/src
$ python main.py
```

4. Copy alamat website pada terminal dan paste pada browser atau click alamat pada terminal.

<br>

## Program Example <a name = "pe"></a>

![](https://i.ibb.co/kK34nsK/contoh.png)