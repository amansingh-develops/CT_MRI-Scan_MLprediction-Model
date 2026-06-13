import React from 'react';
import { Sidebar } from './Sidebar';
import { ChatButton } from '../chatbot/ChatButton';
import { ChatPanel } from '../chatbot/ChatPanel';
import { Outlet } from 'react-router-dom';

export const PageWrapper = () => {
  return (
    <div className="flex h-screen bg-background overflow-hidden selection:bg-primary/30">
      <Sidebar />
      <main className="flex-1 relative flex flex-col min-w-0 h-full overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          <Outlet />
        </div>
      </main>
      <ChatButton />
      <ChatPanel />
    </div>
  );
};
