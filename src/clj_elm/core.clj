(ns clj-elm.core
  (:require [clojure.repl :refer [doc]]
            ;; [clojure.core.typed :as t :only [atom doseq let fn defn ref dotimes defprotocol loop for]]
            ;; [clojure.core.typed :refer :all :exclude [atom doseq let fn defn ref dotimes defprotocol loop for]]
            [incanter.core :as c :exclude [update]]))

;; (ann sign [Num -> Int])
(defn sign
  "The signum function of a real number x."
  ([x]
   {:pre [(number? x)]}
   (cond
     (> x 0) 1
     (= x 0) 0
     :else -1)))

;; (ann ^:no-check clojure.core/float? [Any -> Bool])

;; (ann make-weights [Int -> (Seqable Number)])
(defn make-weights
  "Make a d-dimension-random-feature vector. d is number of dimention of
   feature. All elements are in [-1,1]"
  ([d]
   {:pre [(integer? d)]}
   (take d (repeatedly #(dec (rand 2))))))

;; (ann make-as [Int Int -> (Seqable Number)])
(defn make-ass
  ([d L]
   {:pre [(integer? d) (integer? L)]}
   (take L (repeatedly #(make-weights d)))))

;; (ann make-b [Int -> (Seqable Number)])
(defn make-bs
  ([L]
   {:pre (integer? L)}
   (take L (repeatedly #(first (make-weights 1))))))

(defn standard-sigmoid
  ([x]
   {:pre [(number? x)]
    :post [(number? %)]}
   (/ 1 (+ 1 (Math/exp (- x))))))

(defn g
  ([x]
   (standard-sigmoid x)))

(defn a-hidden-layer-output
  "Return output of hidden-layer_i with xs. As_i is d-dimension. B_i is number.
   Xs is d-dimension."
  ([as_i b_i xs]
   {:pre [(coll? as_i) (number? b_i) (coll? xs)]}
   (g (+ (c/sel (c/mmult (c/trans xs) as_i) 0 0)
         b_i))))

(defn hidden-layer-output-matrix 
  "Ass is d-L-dimension. Xss is d-L-dimesion. Bs is L-dimension."
  ([ass bs xss]
   {:pre [(coll? ass) (coll? (first ass))
          (coll? bs) (number? (first bs))
          (coll? xss) (coll? (first xss))]
    :post [(= (count (first %)) (count ass)) (= (count %) (count xss))]}
   (for [xs_i xss]
     (map #(a-hidden-layer-output %1 %2 xs_i) ass bs))))

(defn pseudo-inverse-matrix
  ([mat]
   {:pre [(coll? mat) (coll? (first mat))]}
   (let [matrix (c/matrix mat)
         n (c/nrow matrix)
         p (c/ncol matrix)]
     (cond
       (= n p) (c/solve matrix)
       (> n p) (c/mmult (c/solve (c/mmult (c/trans matrix)
                                          matrix))
                        (c/trans matrix))
       (< n p) (c/mmult (c/solve (c/mmult matrix
                                          (c/trans matrix)))
                        matrix)))))

(defn output [betas ass bs xs]
  (let [x (c/sum (map #(* %1 (a-hidden-layer-output %2 %3 xs)) betas ass bs))]
    (sign x)))
