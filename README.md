# clj-elm

This is Extreme Learning Machine on Clojure. For only 2 class classification.  

## Usage

`(def dataset (data/read-dataset "data/australian.csv" 14))`

`(def model (train-model australian 100))`

`(predict model (first (:features dataset)))`

Cross validate test:

`(cross-validate dataset 10 100)`

```clojure
Data 0 to 68 .
Data 69 to 137 .
Data 138 to 206 .
Data 207 to 275 .
Data 276 to 344 .
Data 345 to 413 .
Data 414 to 482 .
Data 483 to 551 .
Data 552 to 620 .
Data 621 to 689 .
197/230
```

## License

Copyright © 2015 iyhoo.

Distributed under the Eclipse Public License.
