import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, X, Trash2 } from 'lucide-react';
import { useUiStore } from '../../stores/uiStore';
import { useChatStore } from '../../stores/chatStore';
import { useScanStore } from '../../stores/scanStore';
import { cn } from '../../lib/utils';
import botImage from '../../assets/chatbot-3d-icon-artificial-intelligence_431668-1735.png';

// Simple contextual AI responses based on scan data
const generateResponse = (userMessage, scans) => {
  const msg = userMessage.toLowerCase();
  const latestScan = scans[0];
  const hasAnomaly = scans.some(s => s.analysisState === 'anomaly' || s.result?.hasAnomaly);
  const totalScans = scans.length;
  
  if (msg.includes('result') || msg.includes('scan') || msg.includes('report')) {
    if (!latestScan) {
      return "You don't have any scans uploaded yet. Head to the Upload page to submit your first CT scan for AI analysis.";
    }
    if (latestScan.analysisState === 'anomaly' || latestScan.result?.hasAnomaly) {
      return `Your most recent scan (${latestScan.scanRef || latestScan.scanId}) shows an anomaly detected by our AI model. The system has flagged tissue density patterns that may require clinical review. I recommend discussing these findings with your physician for a comprehensive evaluation.`;
    }
    return `Your most recent scan (${latestScan.scanRef || latestScan.scanId}) appears clear. No significant anomalies were detected by the AI model. This is a positive result, but regular follow-up scans are still recommended.`;
  }
  
  if (msg.includes('stage') || msg.includes('severity') || msg.includes('serious')) {
    return "The staging classification depends on the size, location, and tissue characteristics identified in the segmentation. Based on the current model's analysis, any flagged regions should be discussed with your radiologist for clinical staging. Our AI provides decision support, not a final diagnosis.";
  }
  
  if (msg.includes('what') && (msg.includes('mean') || msg.includes('explain'))) {
    return "The AI segmentation model highlights regions in your scan that have different tissue densities than expected. Orange/red highlights indicate potential areas of concern, while green areas indicate normal tissue. The confidence score tells you how certain the model is about its findings.";
  }
  
  if (msg.includes('next') || msg.includes('do now') || msg.includes('recommend')) {
    if (hasAnomaly) {
      return "Since anomalies were detected, I recommend: 1) Download the full report from the Results page, 2) Schedule a follow-up with your physician, 3) Consider a comparative scan in 3-6 months. Remember, AI analysis is a screening tool — your doctor will provide the definitive assessment.";
    }
    return "Your scans look clear! I recommend continuing with regular screening at the intervals your physician recommends. You can upload new scans anytime for AI analysis.";
  }
  
  if (msg.includes('how many') || msg.includes('total') || msg.includes('count')) {
    return `You currently have ${totalScans} scan(s) in your ScanSight library. ${hasAnomaly ? 'Some scans have flagged anomalies requiring review.' : 'All scans appear clear.'}`;
  }
  
  if (msg.includes('hello') || msg.includes('hi') || msg.includes('hey')) {
    return "Hello! I'm your ScanSight AI assistant. I can help you understand your scan results, explain medical terms, and guide you on next steps. What would you like to know?";
  }
  
  if (msg.includes('confidence') || msg.includes('accuracy')) {
    return "The confidence score represents how certain the AI model is about its prediction. Scores above 85% indicate high confidence. Our model is trained on thousands of clinical CT scans and validated against expert radiologist annotations. However, AI analysis should always be confirmed by a qualified medical professional.";
  }

  return "I can help you with questions about your scan results, explain what the AI findings mean, suggest next steps, or provide general information about the scanning process. Try asking me about your latest scan results or what the findings mean.";
};

export const ChatPanel = () => {
  const { chatOpen, toggleChat } = useUiStore();
  const { messages, addMessage, isTyping, setTyping, clearMessages } = useChatStore();
  const { scans } = useScanStore();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (chatOpen) {
      scrollToBottom();
    }
  }, [messages, chatOpen, isTyping]);

  const handleSend = async (e, textOverride = null) => {
    e?.preventDefault();
    const textToSend = textOverride || input;
    if (!textToSend.trim()) return;

    const userMsg = { role: 'user', text: textToSend, timestamp: Date.now() };
    addMessage(userMsg);
    if (!textOverride) setInput('');
    setTyping(true);

    // Generate contextual response based on actual scan data
    const responseText = generateResponse(textToSend, scans);
    
    // Simulate typing delay (proportional to response length)
    const delay = Math.min(800 + responseText.length * 5, 2500);
    setTimeout(() => {
      setTyping(false);
      addMessage({
        role: 'ai',
        text: responseText,
        timestamp: Date.now(),
      });
    }, delay);
  };

  const quickPrompts = [
    "What does my result mean?",
    "What should I do next?",
    "How many scans do I have?",
  ];

  return (
    <AnimatePresence>
      {chatOpen && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          transition={{ duration: 0.2 }}
          className="fixed bottom-24 right-6 z-40 w-[380px] h-[520px] bg-surface-container-high rounded-xl overflow-hidden flex flex-col shadow-2xl mb-4 backdrop-blur-xl bg-opacity-90 border border-outline-variant/15"
        >
          <style dangerouslySetInnerHTML={{__html: `
            .custom-scrollbar::-webkit-scrollbar { width: 4px; }
            .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
            .custom-scrollbar::-webkit-scrollbar-thumb { background: #3e4850; border-radius: 10px; }
          `}} />

          {/* Header */}
          <div className="h-16 px-5 flex items-center justify-between bg-surface-container-highest border-b border-outline-variant/10 shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-lg shadow-primary/20">
                <img src={botImage} alt="ChatBot" className="w-7 h-7 object-contain drop-shadow-md" />
              </div>
              <div>
                <h3 className="syne font-bold text-on-surface leading-tight">ScanSight AI Doctor</h3>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse"></span>
                  <span className="text-[10px] font-mono text-[#10B981] uppercase tracking-wider font-bold">Online</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <button 
                  onClick={clearMessages} 
                  title="Clear chat"
                  className="text-on-surface-variant hover:text-error transition-colors p-1"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
              <button onClick={toggleChat} className="text-on-surface-variant hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Chat Area */}
          <div className="flex-1 overflow-y-auto p-5 space-y-5 custom-scrollbar">
            {messages.length === 0 && !isTyping && (
              <div className="h-full flex flex-col items-center justify-center text-center px-4">
                <img src={botImage} alt="ChatBot" className="w-16 h-16 mb-4 opacity-50 drop-shadow-lg filter grayscale mix-blend-luminosity" />
                <p className="text-on-surface-variant text-sm">I'm ready to answer questions about your scan results.</p>
              </div>
            )}
            
            {messages.map((msg, idx) => (
              <div key={idx} className={cn("flex flex-col gap-1.5 max-w-[85%]", msg.role === 'user' ? "self-end" : "")}>
                <div className={cn(
                  "p-4 rounded-2xl",
                  msg.role === 'user' 
                    ? "bg-primary rounded-tr-none shadow-md" 
                    : "bg-surface-container-low rounded-tl-none border border-outline-variant/10"
                )}>
                  <p className={cn("text-sm leading-relaxed", msg.role === 'user' ? "text-on-primary font-medium" : "text-on-surface")}>
                    {msg.text}
                  </p>
                </div>
                <span className={cn(
                  "text-[10px] font-mono text-on-surface-variant",
                  msg.role === 'user' ? "text-right mr-1" : "ml-1"
                )}>
                  {msg.timestamp 
                    ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                  }
                </span>
              </div>
            ))}

            {isTyping && (
              <div className="flex flex-col gap-1.5 max-w-[85%]">
                <div className="bg-surface-container-low p-4 rounded-2xl rounded-tl-none border border-outline-variant/10">
                  <div className="flex gap-1 items-center">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/40 animate-bounce"></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/80 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Footer / Quick Prompts */}
          <div className="p-4 bg-surface-container-highest/50 shrink-0">
            <div className="flex flex-wrap gap-2 mb-4">
              {quickPrompts.map((prompt, idx) => (
                <button 
                  key={idx}
                  onClick={(e) => handleSend(e, prompt)}
                  className="text-[11px] px-3 py-1.5 rounded-full bg-surface-container-high border border-outline-variant/20 text-primary hover:bg-primary/10 transition-all text-left"
                >
                  {prompt}
                </button>
              ))}
            </div>
            
            <form onSubmit={(e) => handleSend(e)} className="relative mb-3">
              <input 
                className="w-full h-10 bg-surface-container-lowest border-none rounded-lg text-sm px-4 focus:ring-2 focus:ring-primary/30 transition-all placeholder:text-on-surface-variant/40 outline-none text-on-surface" 
                placeholder="Ask AI Doctor about your scan..." 
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isTyping}
              />
              <button 
                type="submit" 
                disabled={!input.trim() || isTyping}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-primary hover:text-white disabled:opacity-50 disabled:hover:text-primary transition-colors"
              >
                <Send className="w-5 h-5" />
              </button>
            </form>
            
            <p className="text-[9px] text-center text-on-surface-variant/60 leading-tight">
              AI analysis is for clinical decision support and does not constitute a final medical diagnosis. Always consult with a human specialist.
            </p>
          </div>

        </motion.div>
      )}
    </AnimatePresence>
  );
};
