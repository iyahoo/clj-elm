# Profile Results

```clojure
Called by
(def dataset (read-dataset "data/australian.csv" 14 :header true))
(profile :info :Arithmetic (cross-validate dataset 100 10))

15-Nov-11 20:58:37 *** INFO [clj-elm.core] - Profiling: :clj-elm.core/Arithmetic
                                 Id      nCalls       Min        Max       MAD      Mean   Time% Time
:clj-elm.core/pseudo-inverse-matrix          10      6.0s      22.0s      2.9s     10.8s     480 108.4s
:clj-elm.core/a-hidden-layer-output     690,000    51.0μs    316.0ms    55.0μs   114.0μs     347 78.3s
                   :timbre/stats-gc           1     17.4s      17.4s       0ns     17.4s      77 17.4s
     :clj-elm.core/confusion-matrix          10   192.0ms       1.1s   279.0ms   512.0ms      23 5.1s
     :clj-elm.data/ith-feature-list          28     5.0ms     19.0ms     4.0ms     9.0ms       1 250.0ms
         :clj-elm.core/select-count          40    20.0μs     90.0ms     9.0ms     5.0ms       1 202.0ms
            :clj-elm.data/normalise           1    79.0ms     79.0ms       0ns    79.0ms       0 79.0ms
           :clj-elm.data/each-ith-f           2    39.0ms     39.0ms   178.0μs    39.0ms       0 77.0ms
           :clj-elm.core/count-rate         690     3.0μs      5.0ms   101.0μs    65.0μs       0 45.0ms
           :clj-elm.core/update-exp         690     587ns      3.0ms    20.0μs    13.0μs       0 9.0ms
                :clj-elm.core/+-exp           9     3.0μs      1.0ms   213.0μs   134.0μs       0 1.0ms
           :clj-elm.core/exp-result          11     4.0μs    866.0μs   140.0μs   100.0μs       0 1.0ms
       :clj-elm.data/num-of-feature          10     3.0μs     92.0μs    27.0μs    24.0μs       0 242.0μs
         :clj-elm.data/extract-list          20     869ns      9.0μs     2.0μs     4.0μs       0 70.0μs
          :clj-elm.data/remove-list          20     327ns      3.0μs     796ns     1.0μs       0 24.0μs
                         Clock Time                                                          100 22.6s
                     Accounted Time                                                          929 209.9s
```
