# clj-elm

This is Extreme Learning Machine on Clojure. For only 2 class classification.  

## Usage

`$ git clone https://github.com/iyahoo/clj-elm.git`

`$ lein repl`

```clojure
(use 'clj-elm.core)  
(use 'clj-elm.data)  

(def dataset (read-dataset "data/australian.csv" 14 :header true))  
;=> #'clj-elm.core/dataset  

(def model (train-model dataset 20 :norm true))  
;=> #'clj-elm.core/model  

(predict model (first (normalize (:features dataset))))  
;=> -1  
```

This can use dataset CSV or lib-svm form.  

```clojure
;; csv  
(read-dataset path-to-csv-dataset class-column :header true-or-false)  
;; lib-svm  
(read-dataset path-to-lib-svm-form-dataset)  
```

Cross validate test:  

`(cross-validate dataset 10 100)`

## License

Copyright © 2015 iyhoo.

Distributed under the Eclipse Public License.
