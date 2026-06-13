import { create } from 'zustand';

export const useChatStore = create((set) => ({
  messages: [],
  isTyping: false,
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
  setTyping: (bool) => set({ isTyping: bool }),
  clearMessages: () => set({ messages: [], isTyping: false }),
}));
