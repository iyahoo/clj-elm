(ns clj-elm.core
  (:require [clojure.repl :refer [doc]]
            [clj-elm.data :as data]            
            ;; [clojure.core.typed :as t :only [atom doseq let fn defn ref dotimes defprotocol loop for]]
            ;; [clojure.core.typed :refer :all :exclude [atom doseq let fn defn ref dotimes defprotocol loop for]]
            [incanter.core :as c :exclude [update]])
  ;; (:import [clj_elm.data.DataSet])
  )

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
   {:pre [(integer? d)]
    :post [(= (count %) d)]}
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
   (g (+ (c/inner-product as_i xs)
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
       (> n p) (-> matrix
                   (c/trans)
                   (c/mmult matrix)
                   (c/solve)
                   (c/mmult (c/trans matrix)))
       (< n p) (-> matrix
                   (c/mmult (c/trans matrix))
                   (c/solve)
                   (c/mmult matrix))))))

(defrecord Model [ass bs betas])

(defn predict
  ([ass bs betas xs]
   {:pre [(coll? ass) (coll? (first ass)) (coll? bs) (coll? betas) (coll? xs)]}
   (-> (map #(* %1 (a-hidden-layer-output %2 %3 xs)) betas ass bs)
       (c/sum)
       (sign)))
  ([^Model model xs]
   {:pre [(instance? Model model) (coll? xs)]}
   (predict (.ass model) (.bs model) (.betas model) xs)))

(defmulti train-model (fn [dataset _ _] (class dataset)))

(defn train-model [dataset L cidx]
  (let [d (data/num-of-feature dataset)
        ass (make-ass d L)
        bs (make-bs L)
        data (data/data-set dataset cidx)
        xss (c/to-vect (data/normalize (c/to-dataset (.features data))))
        H (hidden-layer-output-matrix ass bs xss)
        T (.classes data)
        betas (c/to-vect (c/mmult (pseudo-inverse-matrix H) T))]
    (Model. ass bs betas)))

;; (defmethod train-model incanter.core.Dataset [dataset L cidx]
;;   (let [d (data/num-of-feature dataset)
;;         ass (make-ass d L)
;;         bs (make-bs L)
;;         data (data/data-set dataset cidx)
;;         xss (c/to-vect (data/normalize (c/to-dataset (.features data))))
;;         H (hidden-layer-output-matrix ass bs xss)
;;         T (.classes data)
;;         betas (c/to-vect (c/mmult (pseudo-inverse-matrix H) T))]
;;     (Model. ass bs betas)))



