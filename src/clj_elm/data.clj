(ns clj-elm.data
  (:require [clojure.repl :refer [doc]]
            [incanter.datasets :as da]
            [incanter.core :as c]
            [incanter.stats :as st]
            [incanter.io :as io]
            [svm.core :as svm]))

(defrecord DataSet [classes features])

(defn ^clojure.lang.PersistentVector parse-lib-svm-data
  ([^clojure.lang.PersistentVector line]
   {:pre [(integer? (first line)) (map? (last line))]}
   (let [[class featmap] line
         fnum (count featmap)
         features (map #(get featmap %) (range 1 (inc fnum)))]
     [class features])))

(defn ^DataSet data-set
  "Make DataSet from incanter-form or csv dataset. Cidx is column number of class."
  ([^incanter.core.Dataset dataset cidx]
   {:pre [(c/dataset? dataset)]}
   (DataSet. (map int (c/to-vect (c/$ :all cidx dataset))) (c/to-vect (c/$ [:not cidx] dataset)))))

(defn ^DataSet read-dataset
  ([^String path ^Integer cidx ^Boolean header]
   {:pre [(string? path) (integer? cidx)]}   
   (-> (io/read-dataset path :delim \, :header header)
       (data-set cidx))))

(defn ^DataSet read-dataset-lib-svm
  ([^String path]
   {:pre [(string? path)]}
   (->> (svm/read-dataset path)
        (map parse-lib-svm-data)
        (#(DataSet. (map first %) (map second %))))))

(defn ^DataSet concat-dataset
  ([^DataSet dsa ^DataSet dsb]
   {:pre [(instance? DataSet dsa) (instance? DataSet dsb)]}
   (DataSet. (concat (:classes dsa) (:classes dsb))
             (concat (:features dsa) (:features dsb)))))

(defn ^DataSet shuffle-dataset
  ([^DataSet dataset]
   {:pre [(instance? DataSet dataset)]}
   (-> (map vector (:classes dataset) (:features dataset))
       (shuffle)
       (#(DataSet. (map first %) (map second %))))))

(defn ^Integer num-of-feature 
  ([^DataSet dataset]
   {:pre [(coll? dataset)]}
   (count (first (:features dataset)))))

(defn ith-feature-list 
  ([^DataSet dataset ^Integer i]
   {:pre [(coll? dataset) (integer? i)]}
   (c/$ i (c/to-dataset dataset))))

(defn ^clojure.lang.PersistentVector each-ith-f 
  ([f ^DataSet dataset]
   (->> (range 0 (count (first dataset)))
        (map #(f (ith-feature-list dataset %)))
        (vec))))

(defn ^clojure.lang.PersistentVector each-ith-mean 
  ([^DataSet dataset]
   {:pre [(coll? dataset)]}
   (each-ith-f st/mean dataset)))

(defn ^clojure.lang.PersistentVector each-ith-sd 
  ([^DataSet dataset]
   {:pre [(coll? dataset)]}
   (each-ith-f st/sd dataset)))

(defn ^clojure.lang.PersistentVector normalize 
  ([^DataSet dataset]
   {:pre [(coll? dataset)]}
   (let [means (each-ith-mean dataset)
         sds (each-ith-sd dataset)]    
     (->> (c/to-vect dataset)
          (map #(map - % means))
          (map #(map / % sds))))))

(defn ^clojure.lang.PersistentVector remove-list
  ([^clojure.lang.PersistentVector lst ^Integer a ^Integer b]
   {:pre [(coll? lst) (integer? a) (integer? b) (> b a)]}
   (concat (take a lst) (drop (inc b) lst))))

(defn ^clojure.lang.PersistentVector extract-list
  ([^clojure.lang.PersistentVector lst ^Integer a ^Integer b]
   {:pre [(coll? lst) (integer? a) (integer? b) (> b a)]}
   (take (inc (- b a)) (drop a lst))))
