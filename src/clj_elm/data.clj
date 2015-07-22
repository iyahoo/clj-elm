(ns clj-elm.data
  (:require [clojure.repl :refer [doc]]
            [incanter.datasets :as da]
            [incanter.core :as c]
            [incanter.stats :as st]
            [incanter.io :as io]
            [svm.core :as svm]))

(defrecord DataSet [classes features])

(defn parse-lib-svm-data [line]
  {:pre [(integer? (first line)) (map? (last line))]}
  (let [[class featmap] line
        fnum (count featmap)
        features (map #(get featmap %) (range 1 (inc fnum)))]
    [class features]))

(defn data-set
  "Make DataSet from incanter-form (csv) dataset. Cidx is column number of class."
  ([^incanter.core.Dataset dataset cidx]
   {:pre [(c/dataset? dataset)]}
   (DataSet. (map int (c/to-vect (c/$ :all cidx dataset))) (c/to-vect (c/$ [:not cidx] dataset)))))

(defn read-dataset
  ([path cidx header]
   {:pre [(string? path) (integer? cidx)]}   
   (-> (io/read-dataset path :delim \, :header header)
       (data-set cidx))))

(defn read-dataset-lib-svm
  ([path]
   {:pre [(string? path)]}
   (->> (svm/read-dataset path)
        (map parse-lib-svm-data)
        (#(DataSet. (map first %) (map second %))))))

(defn concat-dataset [dsa dsb]
  {:pre [(instance? DataSet dsa) (instance? DataSet dsb)]}
  (DataSet. (concat (:classes dsa) (:classes dsb)) (concat (:features dsa) (:features dsb))))

(defn shuffle-dataset
  ([dataset]
   {:pre [(instance? DataSet dataset)]}
   (-> (map vector (:classes dataset) (:features dataset))
       (shuffle)
       (#(DataSet. (map first %) (map second %))))))

(defn num-of-feature 
  ([dataset]
   {:pre [(coll? dataset)]}
   (count (first (:features dataset)))))

(defn get-features 
  ([line]
   {:pre [(coll? line)]}
   (butlast line)))

(defn ith-feature-list 
  ([dataset i]
   {:pre [(coll? dataset) (integer? i)]}
   (c/$ i (c/to-dataset dataset))))

(defn each-ith-f 
  ([f dataset]
   (->> (range 0 (count (first dataset)))
        (map #(f (ith-feature-list dataset %)))
        (vec))))

(defn each-ith-mean 
  ([dataset]
   {:pre [(coll? dataset)]}
   (each-ith-f st/mean dataset)))

(defn each-ith-sd 
  ([dataset]
   {:pre [(coll? dataset)]}
   (each-ith-f st/sd dataset)))

(defn normalize 
  ([dataset]
   {:pre [(coll? dataset)]}
   (let [means (each-ith-mean dataset)
         sds (each-ith-sd dataset)]    
     (->> (c/to-vect dataset)
          (map #(map - % means))
          (map #(map / % sds))))))

(defn remove-list
  ([lst a b]
   {:pre [(coll? lst) (integer? a) (integer? b) (> b a)]}
   (concat (take a lst) (drop (inc b) lst))))

(defn extract-list
  ([lst a b]
   {:pre [(coll? lst) (integer? a) (integer? b) (> b a)]}
   (take (inc (- b a)) (drop a lst))))
