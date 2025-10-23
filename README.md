# FirebaseとNuxtのテンプレートリポジトリ
環境設定の手間を削減するために、よく使っている機能をテンプレートとして設定しました

# 使用技術
- Framework – Nuxt 
- Backend – Firebase
- UI – Tailwind CSS
- Code Quality – ESLint
- Deployment – Vercel

# 環境構築の手段
## 1.リポジトリをクローン
- githubの「このテンプレートを使う」を押下し新たなリポジトリでテンプレートを使用する
- git clone <リポジトリのURL>
## 2. Firebaseの設定
- Firebaseのコンソールからプロジェクトを作成する
- @.envファイルをコピーし.envにリネームする
- .envファイルの環境変数を変更する
- .firebasercのdefaltを使用するproject名に変更する

## 3.パッケージのインストール
- ルート直下でnpm run dev
- functionsディレクトリに移動してnpm installを行う

## 4.ページ起動
- ルート直下でnpm run dev
- functions直下でnpm run serve

## 5.deploy設定
- vercelのページにアクセスし、プロジェクトと接続し環境変数を設定する
