// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: true },
  modules: ["@nuxtjs/tailwindcss"],
  css: ["@/assets/css/common.scss"],
  nitro: {
    preset: "vercel",
  },
  runtimeConfig: {
    public: {
      ENV: process.env[process.env.NODE_ENV + "_" + "ENV"],
      apiKey: process.env[process.env.NODE_ENV + "_" + "apiKey"],
      authDomain: process.env[process.env.NODE_ENV + "_" + "authDomain"],
      projectId: process.env[process.env.NODE_ENV + "_" + "projectId"],
      storageBucket: process.env[process.env.NODE_ENV + "_" + "storageBucket"],
      databaseURL: process.env[process.env.NODE_ENV + "_" + "databaseURL"],
      messagingSenderId:
        process.env[process.env.NODE_ENV + "_" + "messagingSenderId"],
      appId: process.env[process.env.NODE_ENV + "_" + "appId"],
      measurementId: process.env[process.env.NODE_ENV + "_" + "measurementId"],
    },
  },
});
