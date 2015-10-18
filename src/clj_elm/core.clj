(ns clj-elm.core
  (:require [clojure.repl :refer [doc]]
            [clj-elm.data :refer [read-dataset normalize]]
            [clj-elm.data :as data]
            [incanter.core :as c :exclude [update]])
  (:import [clj_elm.data DataSet]))

(defn sign
  "The signum function for a real number x."
  ([x]
   {:pre [(number? x)]}
   (cond
     (> x 0) 1
     (= x 0) 0
     :else -1)))

(defn make-weights
  "Make a d-dimension-random-feature vector. d is number of dimention of
   feature. All elements are in [-1,1]"
  ([d]
   {:pre [(integer? d)]}
   (take d (repeatedly #(dec (rand 2))))))

(defn make-ass
  ([d L]
   {:pre [(integer? d) (integer? L)]}
   (take L (repeatedly #(make-weights d)))))

(defn make-bs
  ([L]
   {:pre (integer? L)}
   (take L (repeatedly #(first (make-weights 1))))))

(defn standard-sigmoid
  ([x]
   {:pre [(number? x)]
    :post [(number? %)]}
   (/ 1 (+ 1 (Math/exp (- x))))))

(defn bipolar-sigmoid
  ([x]
   {:pre [(number? x)]
    :post [(number? %)]}
   (dec (/ 2 (+ 1 (Math/exp (- x)))))))

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
   {:pre [(coll? ass) (coll? (first ass)) (number? (first (first ass)))
          (coll? bs) (number? (first bs))
          (coll? xss) (coll? (first xss))]
    :post [(= (count (first %)) (count ass)) (= (count %) (count xss))]}
   (pmap (fn [xs_i] (pmap #(a-hidden-layer-output %1 %2 xs_i) ass bs)) xss)))

(defn pseudo-inverse-matrix
  ([mat]
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

(defn train-model
  ([dataset L & {:keys [norm] :or {norm false}}]
   ;; {:pre [(instance? DataSet dataset) (integer? L)]}
   (let [normf (if norm data/normalize identity)
         d (data/num-of-feature dataset)
         ass (make-ass d L)
         bs (make-bs L)
         xss (normf (:features dataset))
         H (hidden-layer-output-matrix ass bs xss)
         T (:classes dataset)
         betas (c/to-vect (c/mmult (pseudo-inverse-matrix H) T))]
     (Model. ass bs betas))))

(defn predict  
  ([model xs]
   {:pre [(instance? Model model) (coll? xs)]}
   (predict (:ass model) (:bs model) (:betas model) xs))
  ([ass bs betas xs]
   {:pre [(coll? ass) (coll? (first ass)) (coll? bs) (coll? betas) (coll? xs)]}
   (-> (pmap #(* %1 (a-hidden-layer-output %2 %3 xs)) betas ass bs)
       (c/sum)
       (sign))))

(defn evaluation
  ([results facts]
   {:pre [(coll? results) (coll? facts)]}
   (let [numd (count results)]
     (->> (pmap #(= %1 %2) results facts)
          (filter true?)
          (count)
          (#(/ % numd))))))

(defn _cross-validate
  ([dataset a b L]
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

(defn cross-validate
  ([dataset L k]
   {:pre [(instance? DataSet dataset) (integer? k) (integer? L)]}
   (let [norm-dataset (DataSet. (:classes dataset) (data/normalize (:features dataset)))
         numd (count (:classes norm-dataset)) ; number of data
         groupn (quot numd k)]                ; number of one group's element
     (->> (take k (iterate #(+ % groupn) 0))
          (pmap #(_cross-validate norm-dataset % (+ % (dec groupn)) L))
          (reduce +)          
          (* (/ 1 k))))))
