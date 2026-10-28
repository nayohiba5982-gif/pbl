FROM node:22-alpine

WORKDIR /app

RUN apk add --no-cache libc6-compat

# パッケージ情報をコピーしてインストール
COPY package*.json ./
RUN npm install

# アプリのコードをコピー
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]