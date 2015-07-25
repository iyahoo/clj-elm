(ns clj-elm.core
  (:require [clojure.repl :refer [doc]]
            [clj-elm.data :refer [read-dataset normalize]]
            [clj-elm.data :as data]
            [incanter.core :as c :exclude [update]])
  (:import [clj_elm.data DataSet]))

(defn ^Number sign
  "The signum function for a real number x."
  ([^Number x]
   {:pre [(number? x)]}
   (cond
     (> x 0) 1
     (= x 0) 0
     :else -1)))

(defn ^clojure.lang.PersistentVector make-weights
  "Make a d-dimension-random-feature vector. d is number of dimention of
   feature. All elements are in [-1,1]"
  ([^Integer d]
   {:pre [(integer? d)]}
   (take d (repeatedly #(dec (rand 2))))))

(defn ^clojure.lang.PersistentVector make-ass
  ([^Integer d ^Integer L]
   {:pre [(integer? d) (integer? L)]}
   (take L (repeatedly #(make-weights d)))))

(defn ^clojure.lang.PersistentVector make-bs
  ([^Integer L]
   {:pre (integer? L)}
   (take L (repeatedly #(first (make-weights 1))))))

(defn ^Number standard-sigmoid
  ([^Number x]
   {:pre [(number? x)]
    :post [(number? %)]}
   (/ 1 (+ 1 (Math/exp (- x))))))

(defn bipolar-sigmoid
  ([x]
   {:pre [(number? x)]
    :post [(number? %)]}
   (dec (/ 2 (+ 1 (Math/exp (- x)))))))

(defn ^Number g
  ([^Number x]
   (standard-sigmoid x)))

(defn ^Number a-hidden-layer-output
  "Return output of hidden-layer_i with xs. As_i is d-dimension. B_i is number.
   Xs is d-dimension."
  ([^clojure.lang.PersistentVector as_i ^Number b_i ^clojure.lang.PersistentVector xs]
   {:pre [(coll? as_i) (number? b_i) (coll? xs)]}
   (g (+ (c/inner-product as_i xs)
         b_i))))

(defn ^clojure.lang.PersistentVector hidden-layer-output-matrix
  "Ass is d-L-dimension. Xss is d-L-dimesion. Bs is L-dimension."
  ([^clojure.lang.PersistentVector ass ^Number bs ^clojure.lang.PersistentVector xss]
   {:pre [(coll? ass) (coll? (first ass)) (number? (first (first ass)))
          (coll? bs) (number? (first bs))
          (coll? xss) (coll? (first xss))]
    :post [(= (count (first %)) (count ass)) (= (count %) (count xss))]}
   (pmap (fn [xs_i] (pmap #(a-hidden-layer-output %1 %2 xs_i) ass bs)) xss)))

(defn ^clojure.lang.PersistentVector pseudo-inverse-matrix
  ([^clojure.lang.PersistentVector mat]
   {:pre [(coll? mat) (coll? (first mat))]}
   (let [matrix (c/matrix mat)
         transmat (c/trans matrix)
         n (c/nrow matrix)
         p (c/ncol matrix)]
     (cond
       (= n p) (c/solve matrix)
       (> n p) (-> transmat
                   (c/mmult matrix)
                   (c/solve)
                   (c/mmult transmat))
       (< n p) (-> matrix
                   (c/mmult transmat)
                   (c/solve)
                   (c/mmult matrix))))))

(defrecord Model [ass bs betas])

(defn ^Model train-model
  ([^DataSet dataset ^Integer L & {:keys [^Boolean norm] :or {norm false}}]
   {:pre [(instance? DataSet dataset) (integer? L)]}
   (let [normf (if norm data/normalize identity)
         d (data/num-of-feature dataset)
         ass (make-ass d L)
         bs (make-bs L)
         xss (normf (:features dataset))
         H (hidden-layer-output-matrix ass bs xss)
         T (:classes dataset)
         betas (c/to-vect (c/mmult (pseudo-inverse-matrix H) T))]
     (Model. ass bs betas))))

(defn ^Integer predict  
  ([^Model model ^clojure.lang.PersistentVector xs]
   {:pre [(instance? Model model) (coll? xs)]}
   (predict (:ass model) (:bs model) (:betas model) xs))
  ([^clojure.lang.PersistentVector ass ^clojure.lang.PersistentVector bs
    ^clojure.lang.PersistentVector betas ^clojure.lang.PersistentVector xs]
   {:pre [(coll? ass) (coll? (first ass)) (coll? bs) (coll? betas) (coll? xs)]}
   (-> (pmap #(* %1 (a-hidden-layer-output %2 %3 xs)) betas ass bs)
       (c/sum)
       (sign))))

(defn ^Number evaluation
  ([^clojure.lang.PersistentVector results ^clojure.lang.PersistentVector facts]
   {:pre [(coll? results) (coll? facts)]}
   (let [numd (count results)]
     (->> (pmap #(= %1 %2) results facts)
          (filter true?)
          (count)
          (#(/ % numd))))))

(defn ^Number _cross-validate
  ([^DataSet dataset ^Integer a ^Integer b ^Integer L]
   {:pre [(instance? DataSet dataset) (integer? a) (integer? b) (> b a)]}
   (let [test (DataSet. (data/extract-list (:classes dataset) a b)
                        (data/extract-list (:features dataset) a b))
         sample (DataSet. (data/remove-list (:classes dataset) a b)
                          (data/remove-list (:features dataset) a b))]
     (dorun (println (str "Data " a " to " b ".")))
     (-> (train-model sample L)
         (#(pmap (fn [feature] (predict % feature)) (:features test)))
         (evaluation (:classes test))
         (as-> eva (do (println eva)
                       eva))))))

(defn ^Number cross-validate
  ([^DataSet dataset ^Integer L ^Integer k]
   {:pre [(instance? DataSet dataset) (integer? k) (integer? L)]}
   (let [norm-dataset (DataSet. (:classes dataset) (data/normalize (:features dataset)))
         numd (count (:classes norm-dataset)) ; number of data
         groupn (quot numd k)]                ; number of one group's element
     (->> (take k (iterate #(+ % groupn) 0))
          (pmap #(_cross-validate norm-dataset % (+ % (dec groupn)) L))
          (reduce +)          
          (* (/ 1 k))))))

