import React, { useEffect } from 'react';
import { AppRouter } from './router';
import { useUiStore } from './stores/uiStore';
import { useAuthStore } from './stores/authStore';
import { auth } from './lib/firebase';
import { onAuthStateChanged } from 'firebase/auth';

function App() {
  const { theme } = useUiStore();
  const { setUser, clearUser, setLoading } = useAuthStore();

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
  }, [theme]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      if (firebaseUser) {
        // Use Firebase Auth data directly — no Firestore read needed
        setUser({
          uid: firebaseUser.uid,
          name: firebaseUser.displayName || 'User',
          email: firebaseUser.email,
          role: 'patient', // Default role; can be enhanced via FastAPI endpoint later
        });
      } else {
        clearUser();
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, [setUser, clearUser, setLoading]);

  return <AppRouter />;
}

export default App;
