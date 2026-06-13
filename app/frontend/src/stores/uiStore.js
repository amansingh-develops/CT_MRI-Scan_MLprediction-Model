import { create } from 'zustand';

export const useUiStore = create((set) => ({
  theme: localStorage.getItem('scansight-theme') || 'dark',
  chatOpen: false,
  toggleTheme: () => set((state) => {
    const newTheme = state.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('scansight-theme', newTheme);
    return { theme: newTheme };
  }),
  toggleChat: () => set((state) => ({ chatOpen: !state.chatOpen })),
}));
