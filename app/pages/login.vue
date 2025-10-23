<script setup lang="ts">
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import {
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  updateProfile,
  type User,
} from "firebase/auth";
const { $auth, $ensureUserDoc, $GoogleAuthProvider } = useNuxtApp();
const router = useRouter();

// タブ状態
const mode = ref<"login" | "signup">("login");

// ログイン用
const email = ref("");
const password = ref("");

// サインアップ用
const displayName = ref("");
const emailSign = ref("");
const passwordSign = ref("");
const confirmPassword = ref("");

// UI
const loading = ref(false);
const error = ref("");

// 既ログインならリダイレクト
// onMounted(() => {
//   onAuthStateChanged(auth, (user) => {
//     if (user) router.push(`/${user.uid}`)
//   })
// })

const ensureUserDoc = async (uid: string, user?: User) => {
  await $ensureUserDoc(uid, {
    email: user?.email ?? null,
    displayName: user?.displayName ?? null,
  });
};

// ログイン：Google
const loginWithGoogle = async () => {
  error.value = "";
  loading.value = true;
  try {
    const provider = new $GoogleAuthProvider();
    const result = await signInWithPopup($auth, provider);
    await ensureUserDoc(result.user.uid, result.user);
    //遷移させたいリンクを設定する
    router.push(`/`);
  } catch (err: any) {
    error.value = err?.message || "Googleログインに失敗しました";
  } finally {
    loading.value = false;
  }
};

// ログイン：メール
const loginWithEmail = async () => {
  error.value = "";
  loading.value = true;
  try {
    const result = await signInWithEmailAndPassword(
      $auth,
      email.value,
      password.value,
    );
    await ensureUserDoc(result.user.uid, result.user);
    //遷移させたいリンクを設定する
    router.push(`/`);
  } catch (err: any) {
    error.value = err?.message || "メールログインに失敗しました";
  } finally {
    loading.value = false;
  }
};

// 新規作成：メール
const signUpWithEmail = async () => {
  error.value = "";
  if (!displayName.value.trim()) {
    error.value = "表示名を入力してください";
    return;
  }
  if (!emailSign.value.trim()) {
    error.value = "メールアドレスを入力してください";
    return;
  }
  if (passwordSign.value.length < 6) {
    error.value = "パスワードは6文字以上にしてください";
    return;
  }
  if (passwordSign.value !== confirmPassword.value) {
    error.value = "確認用パスワードが一致しません";
    return;
  }

  loading.value = true;
  try {
    const cred = await createUserWithEmailAndPassword(
      $auth,
      emailSign.value,
      passwordSign.value,
    );
    await updateProfile(cred.user, { displayName: displayName.value });
    await ensureUserDoc(cred.user.uid, cred.user);
    //遷移させたいリンクを設定する
    router.push(`/`);
  } catch (e: any) {
    error.value =
      mapAuthError(e?.code) ?? (e?.message || "アカウント作成に失敗しました");
  } finally {
    loading.value = false;
  }
};

function mapAuthError(code?: string) {
  switch (code) {
    case "auth/email-already-in-use":
      return "このメールアドレスは既に使用されています";
    case "auth/invalid-email":
      return "メールアドレスの形式が正しくありません";
    case "auth/weak-password":
      return "パスワードが弱すぎます（6文字以上を推奨）";
    case "auth/operation-not-allowed":
      return "現在この登録方法は無効です（Firebaseコンソールを確認してください）";
    default:
      return undefined;
  }
}
</script>

<template>
  <div
    class="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center"
  >
    <div
      class="bg-white rounded-2xl shadow-xl p-10 text-center w-full max-w-sm animate-fade-in space-y-5"
    >
      <h1 class="text-2xl font-bold text-gray-800">ようこそ</h1>
      <p class="text-gray-600">
        {{ mode === "login" ? "ログイン" : "新規登録" }}
      </p>

      <!-- Google共通ボタン（ログイン/新規どちらでも可） -->
      <button
        :disabled="loading"
        @click="loginWithGoogle"
        class="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white font-semibold py-2 px-4 rounded-xl w-full transition"
      >
        <!-- Google "G" アイコン -->
        <svg class="w-5 h-5" viewBox="0 0 533.5 544.3" aria-hidden="true">
          <path
            fill="#4285F4"
            d="M533.5 278.4c0-17.4-1.5-34.1-4.4-50.2H272.1v95h147.1c-6.4 34.6-25.9 63.9-55.1 83.5v68h88.9c52-47.9 80.5-118.5 80.5-196.3z"
          />
          <path
            fill="#34A853"
            d="M272.1 544.3c74.6 0 137.1-24.7 182.8-67.4l-88.9-68c-24.7 16.6-56.4 26.4-93.9 26.4-72 0-133-48.6-154.7-114.1H25.7v71.5c45.4 90.5 138.3 151.6 246.4 151.6z"
          />
          <path
            fill="#FBBC05"
            d="M117.4 321.2c-10.7-31.9-10.7-66.3 0-98.2V151.5H25.7c-41.3 82.7-41.3 180.5 0 263.2l91.7-93.5z"
          />
          <path
            fill="#EA4335"
            d="M272.1 107.6c38.6-.6 75.7 13.9 103.9 40.6l77.6-77.6C396.9 24.2 335.6 0 272.1 0 164 0 71.1 61.1 25.7 151.5l91.7 71.5c21.7-65.5 82.7-114.1 154.7-114.1z"
          />
        </svg>
        <span
          >Googleで{{
            mode === "login" ? "ログイン" : "新規登録 / ログイン"
          }}</span
        >
      </button>

      <div class="border-t border-gray-300 my-2"></div>

      <!-- ログインフォーム -->
      <template v-if="mode === 'login'">
        <input
          type="email"
          v-model="email"
          placeholder="メールアドレス"
          class="w-full border border-gray-300 p-2 rounded"
        />
        <input
          type="password"
          v-model="password"
          placeholder="パスワード"
          class="w-full border border-gray-300 p-2 rounded"
        />
        <button
          :disabled="loading"
          @click="loginWithEmail"
          class="flex items-center justify-center gap-2 bg-black hover:bg-gray-800 disabled:opacity-60 text-white font-semibold py-2 px-4 rounded-xl w-full transition"
        >
          <!-- メールアイコン -->
          <svg
            class="w-5 h-5"
            viewBox="0 0 24 24"
            aria-hidden="true"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <rect x="3" y="5" width="18" height="14" rx="2" ry="2"></rect>
            <path d="M3 7l9 6 9-6"></path>
          </svg>
          <span>メールでログイン</span>
        </button>

        <!-- ここがリクエストのポイント：下のリンクでタブ切替 -->
        <p class="text-sm text-gray-600 mt-2">
          はじめての方は
          <button
            class="text-blue-600 hover:underline"
            @click="mode = 'signup'"
          >
            新規作成
          </button>
        </p>
      </template>

      <!-- 新規作成フォーム -->
      <template v-else>
        <input
          type="text"
          v-model="displayName"
          placeholder="表示名"
          class="w-full border border-gray-300 p-2 rounded"
        />
        <input
          type="email"
          v-model="emailSign"
          placeholder="メールアドレス"
          class="w-full border border-gray-300 p-2 rounded"
        />
        <input
          type="password"
          v-model="passwordSign"
          placeholder="パスワード（6文字以上）"
          class="w-full border border-gray-300 p-2 rounded"
        />
        <input
          type="password"
          v-model="confirmPassword"
          placeholder="パスワード（確認）"
          class="w-full border border-gray-300 p-2 rounded"
        />
        <button
          :disabled="loading"
          @click="signUpWithEmail"
          class="bg-black hover:bg-gray-800 disabled:opacity-60 text-white font-semibold py-2 px-4 rounded-xl w-full transition"
        >
          アカウントを作成
        </button>

        <p class="text-sm text-gray-600 mt-2">
          すでにアカウントをお持ちですか？
          <button class="text-blue-600 hover:underline" @click="mode = 'login'">
            ログインへ
          </button>
        </p>
      </template>

      <p v-if="error" class="text-red-600 text-sm mt-2">{{ error }}</p>
    </div>
  </div>
</template>

<style scoped>
@keyframes fade-in {
  0% {
    opacity: 0;
    transform: translateY(20px);
  }
  100% {
    opacity: 1;
    transform: translateY(0);
  }
}
.animate-fade-in {
  animation: fade-in 0.8s ease-out;
}
</style>
