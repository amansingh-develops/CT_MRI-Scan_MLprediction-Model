import React from 'react';
import { Bot, X } from 'lucide-react';
import { useUiStore } from '../../stores/uiStore';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../../lib/utils';
import botImage from '../../assets/chatbot-3d-icon-artificial-intelligence_431668-1735.png';

export const ChatButton = () => {
  const { chatOpen, toggleChat } = useUiStore();

  return (
    <motion.button
      animate={!chatOpen ? { 
        y: [0, -6, 0],
        boxShadow: [
          "0px 0px 15px rgba(14, 165, 233, 0.4)",
          "0px 0px 35px rgba(14, 165, 233, 0.9)",
          "0px 0px 15px rgba(14, 165, 233, 0.4)"
        ]
      } : { y: 0, boxShadow: "0px 0px 0px rgba(14, 165, 233, 0)" }}
      transition={!chatOpen ? { duration: 3, repeat: Infinity, ease: "easeInOut" } : { duration: 0.2 }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={toggleChat}
      className={cn(
        "fixed top-4 right-4 z-50 flex h-14 w-14 items-center justify-center rounded-full relative group transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background",
        chatOpen ? "bg-surface-container-highest text-on-surface" : "bg-gradient-to-t from-primary-container to-primary"
      )}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={chatOpen ? 'close' : 'open'}
          initial={{ opacity: 0, rotate: -90 }}
          animate={{ opacity: 1, rotate: 0 }}
          exit={{ opacity: 0, rotate: 90 }}
          transition={{ duration: 0.15 }}
          className={cn(
            "flex items-center justify-center w-full h-full text-on-primary",
            chatOpen ? "text-on-surface" : "text-on-primary group-hover:rotate-12 transition-transform"
          )}
        >
          {chatOpen ? <X className="h-6 w-6" /> : <img src={botImage} alt="ChatBot" className="h-9 w-9 object-contain" />}
        </motion.div>
      </AnimatePresence>

      {/* Notification Badge */}
      {!chatOpen && (
        <div className="absolute -top-1 -right-1 w-5 h-5 bg-tertiary-container rounded-full border-2 border-background flex items-center justify-center shadow-lg">
          <span className="text-[10px] font-black text-on-tertiary-container">1</span>
        </div>
      )}

      {/* Animated Subtle Glow */}
      {!chatOpen && (
        <motion.div 
          animate={{ scale: [1.2, 1.7, 1.2], opacity: [0.4, 0.8, 0.4] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          className="absolute inset-0 rounded-full bg-primary/40 blur-xl -z-10"
        />
      )}
    </motion.button>
  );
};
