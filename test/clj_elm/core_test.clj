(ns clj-elm.core-test
  (:require [clj-elm.core :refer :all]
            [clj-elm.data :as data :refer [normalize data-set]]
            [midje.sweet :refer :all]
            [midje.repl :refer (autotest load-facts)]
            [incanter.core :as c :exclude [update]]
            [incanter.io :as io]
            [svm.core :as svm])
  (:import [clj_elm.data DataSet]))

(def australian
  (data/read-dataset "data/australian.csv" 14 :header true))

(def libsvmlian
  (data/read-dataset-lib-svm "data/australian"))

(facts "test-sign"
  (fact "(sign x)"
    (sign 100)      =>  1
    (sign 0.1)      =>  1
    (sign -100)     => -1
    (sign -0.00001) => -1
    (sign 0)        =>  0))

(facts "test-reverse-sign"
  (fact "(reverse-sign x)"
    (reverse-sign 100)      => -1
    (reverse-sign 0.1)      => -1
    (reverse-sign -100)     =>  1
    (reverse-sign -0.00001) =>  1
    (reverse-sign 0)        =>  0))

(facts "test-make-weights"
  (fact "(make-weights d)"
    (reduce #(and %1 %2)
            (map #(< -1.0 % 1.0)
                 (make-weights 100)))
    => true))

(facts "test-make-ass"
  (fact "(make-ass d L)"
    (count (make-ass 10 3)) => 3
    (count (first (make-ass 10 3))) => 10))

(facts "test-make-bs"
  (fact "(make-bs L)"
    (count (make-bs 3)) => 3))

(facts "test-standard-sigmoid"
  (fact "(standard-sigmoid x)"
    (standard-sigmoid 0) => 0.5
    (filter #(> % 1.0) (map standard-sigmoid (take 10 (repeatedly #(rand 100)))))
    => []))

(facts "test-a-hidden-layer-output"
  (fact "(a-hidden-layer-output as_i b_i xs)"
    (a-hidden-layer-output [0 0 0] 0 [0 0 0]) => 0.5))

(facts "test-hidden-layer-output-matrix"
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

(facts "test-make-double-jarray"
  (fact "(make-double-jarray array2h)"))

(defn roundn [^Double x ^Integer n]
  (-> (BigDecimal. x)
      (.setScale n BigDecimal/ROUND_HALF_UP)))

(facts "test-pseudo-inverse-matrix"
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
    (let [model (train-model australian 20 :norm true)]
      (predict model (first (normalize (:features australian))))
      => -1)
    (let [model (train-model libsvmlian 20 :norm true)]
      (predict model (first (normalize (:features libsvmlian))))
      => -1))
  (fact "(binding [*sign-reverse* true] (train-model dataset l))"
    (binding [*sign-reverse* true]
      (let [model (train-model australian 20 :norm true)]
        (predict model (first (normalize (:features australian))))
        => 1)
      (let [model (train-model libsvmlian 20 :norm true)]
        (predict model (first (normalize (:features libsvmlian))))
        => 1))))

(facts "test-update-exp"
  (fact "(update-exp fn key exp) return reference of exp"
    (let [exp {:TP 0 :FP 0}]
      (update-exp inc :TP exp)
      => {:TP 1 :FP 0})))

(facts "test-count-rate"
  (fact "(count-rate pred fact exp)"
    (let [exp {:TP 0 :FP 0 :TN 10 :FN 10}]
      (:TP (count-rate 1 1 exp))
      => 1
      (:FP (count-rate 1 -1 exp))
      => 1
      (:TN (count-rate -1 -1 exp))
      => 11
      (:FN (count-rate -1 1 exp))
      => 11)))

(facts "test-exp-result"
  (fact "(exp-result exp)"
    (exp-result {:TP 10 :FP 10 :TN 10 :FN 10})
    => {:TP 10 :FP 10 :TN 10 :FN 10
        :Accuracy 1/2 :Recall 1/2 :Precision 1/2}))

(facts "test-confusion-matrix"
  (fact "(confusion-matrix preds facts exp)"
    (let [exp {:TP 0 :FP 0 :TN 0 :FN 0}]
      (confusion-matrix [1 1 -1 -1] [1 1 -1 -1] exp))
    => {:TP 2, :FP 0, :TN 2, :FN 0, :Accuracy 1, :Recall 1, :Precision 1}))

(facts "test-evaluation"
  (facts "(evaluation results facts)"
    (let [dummy {:TP 0 :FP 0 :TN 0 :FN 0}]
      (:Accuracy (evaluation [1 1 1 1 1 1 1 1 1 1] [1 1 1 1 1 1 1 1 1 1] dummy)))
    => 1
    (let [dummy {:TP 0 :FP 0 :TN 0 :FN 0}]
      (:Accuracy (evaluation [-1 -1 -1 -1 -1 1 1 1 1 1] [1 1 1 1 1 1 1 1 1 1] dummy)))
    => 1/2
    (let [dummy {:TP 0 :FP 0 :TN 0 :FN 0}]
      (:Accuracy (evaluation [-1 -1 -1 -1 -1 -1 -1 -1 -1 1] [1 1 1 1 1 1 1 1 1 1] dummy)))
    => 1/10))

(facts "test-select-count"
  (fact "(select-count cond dataset)"
    (select-count odd? [1 2 3 4 5])
    => 3
    (select-count #(= % 1) (:classes (DataSet. [1 1 -1 -1] [[1 1] [1 1] [1 1] [1 1]])))
    => 2))

(facts "test-_print-exp-data"
  (fact "(_print-exp-data exp)"
    (_print-exp-data {:L 10
                      :length-train-negative 10 :length-train-positive 10
                      :length-test-negative 10 :length-test-positive 10
                      :TP 10 :FP 10 :TN 10 :FN 10
                      :Accuracy 1/2 :Recall 1/2 :Precision 1/2})
    => "L:10\n length(train_negative): 10\n length(test_negative): 10\n length(train_positive): 10\n length(test_positive): 10\n TP: 10, FP: 10, TN: 10, FN: 10\n Accuracy: 1/2,Recall: 1/2,Precision: 1/2\n\n"))

(facts "test-+-exp"
  (fact "(+-exp exp1 exp2)"
    (+-exp {:L 500
            :length-train-negative 100 :length-test-negative 10 :length-train-positive 100 :length-test-positive 10
            :TP 10 :FP 10 :TN 10 :FN 10}
           {:L 500
            :length-train-negative 100 :length-test-negative 10 :length-train-positive 100 :length-test-positive 10
            :TP 10 :FP 10 :TN 10 :FN 10})
    => {:L 500
        :length-train-negative 200 :length-test-negative 20 :length-train-positive 200 :length-test-positive 20
        :TP 20 :FP 20 :TN 20 :FN 20}))
