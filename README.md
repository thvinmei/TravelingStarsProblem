# TSP1StStars
巡回セールスマン座

## 発端
[なーと氏のTweet](https://twitter.com/Canopacher/status/1368499997826748417)
>巡回セールスマン座（全天の6等級以下の星全てを最短ルートで結ぶ）

## 実装
### 参照するデータ
ヒッパルコス衛星のデータを利用しています。
[ヒッパルコス星表|Astro Commons](http://astronomy.webcrow.jp/hip/)で配布されている、[hip_lite_major.csv](http://astronomy.webcrow.jp/hip/hip_lite_major.csv)と[hip_proper_name.csv](http://astronomy.webcrow.jp/hip/hip_proper_name.csv)を読み込みます。

### 巡回セールスマン問題の解法
[巡回セールスマン問題 | Code Craft House
](http://codecrafthouse.jp/p/2016/08/traveling-salesman-problem/)のプログラムを改変して使っています。