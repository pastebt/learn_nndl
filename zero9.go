package main

import (
    "os"
    "log"
    "math"
    "bufio"
    "strconv"
    "strings"
    "github.com/gonum/matrix/mat64"
)


func load_one(infn string) (err error) {
    fin, err := os.Open(infn)
    if err != nil {
        log.Fatal(err)
    }
    defer fin.Close()
    println(infn)
    scanner := bufio.NewScanner(fin)
    for scanner.Scan() {
        cols := strings.Fields(scanner.Text())
        if len(cols) != 785 { log.Fatalf("got column %d", len(cols)) }
        var n int64
        n, err = strconv.ParseInt(cols[0], 10, 32)
        if err != nil { log.Fatal(cols[0], err) }
        dx := make([]float64, 784)
        var f float64
        for i := 1; i < 785; i++ {
            f, err = strconv.ParseFloat(cols[i], 32)
            if err != nil { log.Fatal(cols[i], err) }
            dx[i - 1] = f
        }
        m := mat64.NewDense(len(dx), 1, dx)
        if n > 9 || f > 10000 || m == nil { log.Fatalf("n=%d, f=%f, m=%#v", n, f, m) }
    }
    return
}


// The sigmoid function.
func sigmoid(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}


// Derivative of the sigmoid function.
func sigmoid_prime(z float64) float64 {
    return sigmoid(z) * (1 - sigmoid(z))
}


type Network struct {
    num_layers int
    biases  []*mat64.Dense
    weights []*mat64.Dense
}


func NewNetwork(sizes []int) *Network {
    n := &Network{num_layers: len(sizes)}
    return n
}

func main() {
    load_one("trai_data.txt")
    load_one("vali_data.txt")
    load_one("test_data.txt")
}
