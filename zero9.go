package main

import (
    "os"
    "log"
    "math"
    "math/rand"
    "bufio"
    "strconv"
    "strings"
    "github.com/gonum/floats"
//    "github.com/gonum/blas/cgo"
    "github.com/gonum/blas/blas64"
    "github.com/gonum/matrix/mat64"
)


type ITEM struct {
    x, yv *mat64.Dense
    yi int
}


func load_one(infn string) (dat []*ITEM, err error) {

//    blas64.Use(cgo.Implementation{})

    fin, err := os.Open(infn)
    if err != nil {
        log.Fatal(err)
    }
    defer fin.Close()
    println(infn)
    dat = make([]*ITEM, 0, 1000)
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
        if n > 9 || f > 10000 || m == nil {
            log.Fatalf("n=%d, f=%f, m=%#v", n, f, m)
        }
        v := mat64.NewDense(10, 1, nil)
        y := int(n)
        v.Set(y, 0, 1.0)
        dat = append(dat, &ITEM{x: m, yv: v, yi: y})
    }
    return
}


// The sigmoid function.
func sigmoid_(z *mat64.Dense) *mat64.Dense {
    //return 1.0 / (1.0 + math.Exp(-z))
    ret := mat64.NewDense(0, 0, nil)
    ret.Apply(func(x, y int, v float64) float64 {
                return 1.0 / (1.0 + math.Exp(-v))
              }, z)
    return ret
}

func sigmoid64(fs []float64) {
    for i, f := range fs {
        fs[i] = 1.0 / (1.0 + math.Exp(-f))
        // http://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
        // bad, not even close
        //fs[i] = math.Tanh(f)
        //fs[i] = f / (1 + math.Abs(f))
    }
}
func sigmoid(z *mat64.Dense) *mat64.Dense {
    s := mat64.DenseCopyOf(z)
    sigmoid64(s.RawMatrix().Data)
    return s
}


// Derivative of the sigmoid function.
func sigmoid_prime_(z *mat64.Dense) *mat64.Dense {
    //return sigmoid(z) * (1 - sigmoid(z))
    s := sigmoid(z)
    m := mat64.NewDense(0, 0, nil)
    m.Apply(func(x, y int, v float64)float64{
                return v * (1.0 - v)
            }, s)
    return m
}

func sigmoid_prime(z *mat64.Dense) *mat64.Dense {
    s := mat64.DenseCopyOf(z)
    fs := s.RawMatrix().Data
    sigmoid64(fs)
    for i, f := range fs {
        fs[i] = f * (1.0 - f)
    }
    return s
}


func randyx(y, x int, d float64) *mat64.Dense {
    z := x * y
    dat := make([]float64, z)
    for i := 0; i < z; i++ {
        //dat[i] = rand.Float64() * 2.0 - 1.0
        dat[i] = rand.NormFloat64() / d
    }
    return mat64.NewDense(y, x, dat)
}


func npdot(w, a mat64.Matrix) *mat64.Dense {
    z := mat64.NewDense(0, 0, nil)
    z.Product(w, a)
    return z
}


func argmax(a *mat64.Dense) int {
    i := floats.MaxIdx(a.RawMatrix().Data)
    return i
}


// http://stackoverflow.com/questions/12264789/shuffle-array-in-go
func shuffle(dat []*ITEM) {
    for i := range dat {
        j := rand.Intn(i + 1)
        dat[i], dat[j] = dat[j], dat[i]
    }
}


func zeros(ts []*mat64.Dense) []*mat64.Dense {
    zs := make([]*mat64.Dense, len(ts))
    for i, t := range ts {
        r, c := t.Dims()
        zs[i] = mat64.NewDense(r, c, nil)
    }
    return zs
}


func add(dst, src *mat64.Dense) (z *mat64.Dense) {
    rd, cd := dst.Dims()
    rs, cs := src.Dims()
    if rd == rs && cd == cs {
        z = mat64.NewDense(0, 0, nil)
        z.Add(dst, src)
        return
    }
    if (rd == 1 || cd == 1) {
        src, dst = dst, src
        rd, cd, rs, cs = rs, cs, rd, cd
    }
    if rs == 1 && cs == cd {
        vs := src.RowView(0).RawVector()
        z = mat64.DenseCopyOf(dst)
        for r := 0; r < rd; r++ {
            vd := z.RowView(r).RawVector()
            blas64.Axpy(cd, 1, vs, vd)
        }
        return
    } else if cs == 1 && rd == rs {
        vs := src.ColView(0).RawVector()
        z = mat64.DenseCopyOf(dst)
        for c := 0; c < cd; c++ {
            vd := z.ColView(c).RawVector()
            blas64.Axpy(rd, 1, vs, vd)
        }
        return
    }
    return
}


func main() {
    trai, err := load_one("trai_data.txt")
    if err != nil { log.Fatal(err) }
    //_, err := load_one("vali_data.txt")
    //if err != nil { log.Fatal(err) }
    test, err := load_one("test_data.txt")
    if err != nil { log.Fatal(err) }
    n := NewNetwork([]int{784, 30, 10})
    n.SGD(trai, 30, 10, 4.0, test)
}
