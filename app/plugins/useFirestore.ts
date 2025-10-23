import { getAuth, type Auth, GoogleAuthProvider } from "firebase/auth";
import {
  getFirestore,
  type Firestore,
  doc,
  getDoc,
  setDoc,
  serverTimestamp,
} from "firebase/firestore";
export default defineNuxtPlugin(() => {
  const auth: Auth = getAuth();
  const db: Firestore = getFirestore();

  const ensureUserDoc = async (
    uid: string,
    user?: { email: string | null; displayName: string | null },
  ) => {
    const userRef = doc(db, "users", uid);
    const snap = await getDoc(userRef);
    if (!snap.exists()) {
      await setDoc(userRef, {
        createdAt: serverTimestamp(),
        email: user?.email ?? null,
        displayName: user?.displayName ?? null,
      });
    } else {
      await setDoc(
        userRef,
        {
          updatedAt: serverTimestamp(),
          email: user?.email ?? null,
          displayName: user?.displayName ?? null,
        },
        { merge: true },
      );
    }
  };

  return {
    provide: {
      auth,
      db,
      ensureUserDoc,
      GoogleAuthProvider,
    },
  };
});
