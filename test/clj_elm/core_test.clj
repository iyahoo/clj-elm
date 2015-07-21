(ns clj-elm.core-test
  (:require [clj-elm.core :refer :all]
            [clj-elm.data :as data :refer [normalize get-features data-set]]
            ;; [clojure.core.typed :as t :only [atom doseq let fn defn ref dotimes defprotocol loop for]]
            ;; [clojure.core.typed :refer :all :exclude [atom doseq let fn defn ref dotimes defprotocol loop for]]
            [midje.sweet :refer :all]
            [midje.repl :refer (autotest load-facts)]
            [incanter.core :as c :exclude [update]]
            [incanter.io :as io]
            [svm.core :as svm]))

(def australian
  (data/read-dataset "data/australian.csv" 14))

(def libsvmlian
  (data/read-dataset-lib-svm "data/australian"))

(facts "test-sign"
  (fact "(sign x)"
    (sign 100)      =>  1
    (sign 0.1)      =>  1
    (sign -100)     => -1
    (sign -0.00001) => -1
    (sign 0)        =>  0))

(facts "test-make-weights"
  (fact "(make-weights d)"
    (map #(< -1.0 % 1.0)
         (make-weights 100))
    => (take 100 (repeat true))))

(facts "test-make-ass"
  (fact "(make-ass d L)"
    (count (make-ass 10 3)) => 3
    (count (first (make-ass 10 3))) => 10))

(fact "test-make-bs"
  (fact "(make-bs L)"
    (count (make-bs 3)) => 3))

(facts "test-standard-sigmoid"
  (fact "(standard-sigmoid x)"
    (standard-sigmoid 0) => 0.5))

(fact "test-a-hidden-layer-output"
  (fact "(a-hidden-layer-output as_i b_i xs)"
    (a-hidden-layer-output [0 0 0] 0 [0 0 0]) => 0.5))

(fact "test-hidden-layer-output-matrix"
  (fact "(hidden-layer-output-matrix ass bs xss)"
    (hidden-layer-output-matrix
     [[0 0 0] [0 0 0] [0 0 0]]
     [0 0 0]
     [[0 0 0] [0 0 0] [0 0 0]])
    => [[0.5 0.5 0.5]
        [0.5 0.5 0.5]
        [0.5 0.5 0.5]]

    (hidden-layer-output-matrix
     [[0 0 0 0] [0 0 0 0] [0 0 0 0]]
     [0 0 0]
     [[0 0 0 0] [0 0 0 0] [0 0 0 0]])
    => [[0.5 0.5 0.5]
        [0.5 0.5 0.5]
        [0.5 0.5 0.5]]))

(defn roundn [^Double x ^Integer n]
  (-> (BigDecimal. x)
      (.setScale n BigDecimal/ROUND_HALF_UP)))

(fact "test-pseudo-inverse-matrix"
  (fact "(pseudo-inverse-matrix mat)"
    (let [A [[2 0 0]
             [0 2 0]
             [0 0 2]]
          B [[ 50   7 75]
             [0.5 0.2  3]
             [ 50   0 20]
             [5.0 6.0 50]]
          ;; http://cis-jp.blogspot.jp/2012/08/blog-post_10.html
          C [[ 5 20  3]
             [10 77 60]
             [ 8  3 70]
             [ 9 11 28]]]
      (c/matrix (c/mmult (pseudo-inverse-matrix A) A))
      => (c/identity-matrix 3)
      (c/matrix (c/matrix-map #(roundn % 10)
                              (c/mmult (pseudo-inverse-matrix B) B)))
      => (c/identity-matrix 3)
      (c/matrix (c/matrix-map #(roundn % 10)
                              (c/mmult (pseudo-inverse-matrix (c/trans B)) B)))
      => (c/identity-matrix 3)
      (c/matrix (c/matrix-map #(roundn % 10)
                              (c/mmult (pseudo-inverse-matrix (c/trans C)) C)))
      => (c/identity-matrix 3))))

(facts "test-train-model-and-predict"
  (fact "(train-model dataset l) (predict model xs)"
    (let [model (train-model australian 20)]
      (predict model (first (normalize (:features australian))))
      => -1)
    (let [model (train-model libsvmlian 20)]
      (predict model (first (normalize (:features libsvmlian))))
      => -1)))

(facts "test-evaluation"
  (facts "(evaluation results facts)"
    (evaluation [1 1 1 1 1 1 1 1 1 1] [1 1 1 1 1 1 1 1 1 1])
    => 1
    (evaluation [-1 -1 -1 -1 -1 1 1 1 1 1] [1 1 1 1 1 1 1 1 1 1])
    => 1/2
    (evaluation [-1 -1 -1 -1 -1 -1 -1 -1 -1 1] [1 1 1 1 1 1 1 1 1 1])
    => 1/10))
