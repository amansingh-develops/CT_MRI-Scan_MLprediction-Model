import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Microscope, ArrowRight, CloudUpload, Bot, LineChart } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuthStore } from '../stores/authStore';

const Landing = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="min-h-screen bg-[#070B14] text-on-surface selection:bg-primary selection:text-on-primary font-body overflow-x-hidden">
      
      {/* Top Navigation Bar */}
      <header className="fixed top-0 left-0 right-0 z-50 flex justify-between items-center px-6 py-3 h-16 bg-[#0f131d]/90 backdrop-blur-md border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary-container flex items-center justify-center">
            <Microscope className="text-on-primary w-5 h-5" />
          </div>
          <span className="syne font-black text-[#89ceff] tracking-tighter text-xl">ScanSight</span>
        </div>
        
        <nav className="hidden md:flex items-center gap-8">
          <Link className="syne font-bold text-[#89ceff] text-sm" to="/dashboard">Dashboard</Link>
          <Link className="text-slate-400 hover:bg-[#262a34] transition-colors duration-200 px-3 py-1 rounded-md text-sm" to="/upload">Upload</Link>
          <Link className="text-slate-400 hover:bg-[#262a34] transition-colors duration-200 px-3 py-1 rounded-md text-sm" to="/dashboard">Pricing</Link>
        </nav>

        <div className="flex items-center gap-4">
          {isAuthenticated ? (
            <Link 
              to="/dashboard" 
              className="bg-gradient-to-b from-[#89ceff] to-[#0ea5e9] border-t border-white/20 text-on-primary px-5 py-2 rounded-xl text-sm font-bold shadow-[0_0_20px_rgba(137,206,255,0.3)] transform hover:brightness-110 active:scale-95 transition-all"
            >
              Go to Dashboard
            </Link>
          ) : (
            <>
              <Link to="/signin" className="text-on-surface-variant hover:text-on-surface text-sm font-medium transition-colors">
                Sign In
              </Link>
              <Link 
                to="/signup" 
                className="bg-gradient-to-b from-[#89ceff] to-[#0ea5e9] border-t border-white/20 text-on-primary px-5 py-2 rounded-xl text-sm font-bold shadow-[0_0_20px_rgba(137,206,255,0.3)] transform hover:brightness-110 active:scale-95 transition-all"
              >
                Get Started
              </Link>
            </>
          )}
        </div>
      </header>

      <main className="relative pt-16 overflow-hidden">
        {/* Hero Section */}
        <section 
          className="relative min-h-screen flex flex-col items-center justify-center px-6"
          style={{
            background: 'radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.15) 0%, rgba(7, 11, 20, 0) 70%)'
          }}
        >
          {/* Background Blobs */}
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 blur-[120px] rounded-full pointer-events-none"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary-container/10 blur-[120px] rounded-full pointer-events-none"></div>
          
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="relative z-10 text-center max-w-4xl mx-auto mt-20"
          >
            <h1 className="font-['Plus_Jakarta_Sans'] font-light text-6xl md:text-8xl tracking-tight leading-[1.1] mb-8">
              See What <span className="font-extrabold italic text-[#0EA5E9]">Others</span> Miss
            </h1>
            <p className="font-body text-on-surface-variant text-xl md:text-2xl mb-12 max-w-2xl mx-auto leading-relaxed">
              AI-powered liver CT scan analysis. Upload your scan. Get results in seconds with clinical precision.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
              <Link 
                to="/signup"
                className="bg-gradient-to-b from-[#89ceff] to-[#0ea5e9] border-t border-white/20 text-on-primary px-8 py-4 rounded-xl text-lg font-bold w-full sm:w-auto flex items-center justify-center gap-2 hover:brightness-110 transition-all shadow-[0_0_30px_rgba(137,206,255,0.2)]"
              >
                Get Started — Patient
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link 
                to="/signup"
                className="bg-surface-container-high hover:bg-surface-bright text-on-surface px-8 py-4 rounded-xl text-lg font-bold w-full sm:w-auto transition-colors"
              >
                I'm a Doctor →
              </Link>
            </div>
          </motion.div>

          {/* Analysis Screen Mockup */}
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.3 }}
            className="mt-20 relative w-full max-w-5xl group z-20 mb-20"
          >
            <div className="absolute -inset-1 bg-gradient-to-r from-primary to-secondary opacity-30 blur-xl transition duration-1000 group-hover:opacity-50"></div>
            
            <div className="relative rounded-2xl overflow-hidden border border-white/10 shadow-2xl" style={{ backdropFilter: 'blur(12px)', background: 'rgba(27, 32, 41, 0.7)' }}>
              {/* Window Header */}
              <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5 bg-surface-container-lowest">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-error/50"></div>
                  <div className="w-3 h-3 rounded-full bg-tertiary/50"></div>
                  <div className="w-3 h-3 rounded-full bg-secondary/50"></div>
                </div>
                <div className="mx-auto text-xs font-mono text-outline tracking-widest">SCAN_ANALYSIS_V2.0.4.DICOM</div>
              </div>

              {/* Mockup Content */}
              <div className="grid grid-cols-1 md:grid-cols-12 h-auto md:h-[500px]">
                {/* Left image area */}
                <div className="md:col-span-8 bg-black relative h-64 md:h-auto">
                  <img 
                    alt="Analysis UI" 
                    className="w-full h-full object-cover opacity-80" 
                    src="https://lh3.googleusercontent.com/aida-public/AB6AXuAwLYu5s_Z4jcj-2tIVt255jl0O4YfR7iDPr5ekGqSU0CkfA8t_bKIQBL-q8ZwaGA1St-aIk6-1wz3Z8nbzbEARiiq5hHPZaZXk6gsZ3lCyE3JJ-A7FrzuD8MZ1kdJP6ue9CsVRD8Mduh1nUA590JckFYJLUC7a9xESecThm6vQwgF8pqQi_0SOUv5AsxJUwvOjqrfNsKv0x11aI-L_LMHaNgWCKDSGr59JEtN3qoD2oTf_Q7DG2v7Vqy4G4b0u5reo8KnT2BgmMOc"
                  />
                  {/* Overlay Grid */}
                  <div className="absolute inset-0 grid grid-cols-8 grid-rows-8 pointer-events-none">
                    {Array.from({ length: 64 }).map((_, i) => (
                      <div key={i} className="border-[0.5px] border-white/5"></div>
                    ))}
                  </div>
                </div>

                {/* Right stats area */}
                <div className="md:col-span-4 bg-surface-container flex flex-col p-6 gap-6">
                  <div className="space-y-2">
                    <span className="font-mono text-[10px] text-primary tracking-widest uppercase">Analysis Status</span>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-secondary shadow-[0_0_8px_#4fdbc8] animate-pulse"></div>
                      <span className="text-sm font-bold">COMPLETED</span>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="p-4 rounded-xl bg-surface-container-high space-y-1">
                      <span className="text-[10px] text-on-surface-variant font-medium">LIVER VOLUME</span>
                      <div className="text-2xl font-bold font-mono text-on-surface">1,542 <span className="text-xs text-outline font-sans">ml</span></div>
                    </div>
                    
                    <div className="p-4 rounded-xl bg-surface-container-high space-y-1 border-l-4 border-error">
                      <span className="text-[10px] text-error font-bold">ANOMALY DETECTED</span>
                      <div className="text-sm font-medium text-on-surface">Region of interest found in Segment VIII</div>
                    </div>
                  </div>
                  
                  <button className="mt-auto w-full py-3 bg-primary/10 hover:bg-primary/20 transition-colors border border-primary/20 text-primary rounded-lg text-sm font-bold">
                    Export Report
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Features Section */}
        <section className="py-24 px-6 max-w-7xl mx-auto relative z-20 bg-[#070B14]">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="p-8 rounded-2xl bg-[#0D1424] border border-white/5 group hover:bg-[#151f33] transition-all duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
                <CloudUpload className="text-primary w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold syne mb-4 text-on-surface">Upload & Analyze</h3>
              <p className="text-on-surface-variant leading-relaxed text-sm">Securely upload DICOM files directly from your device. Our engine processes them with HIPAA-compliant encryption.</p>
            </motion.div>
            
            {/* Feature 2 */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="p-8 rounded-2xl bg-[#0D1424] border border-white/5 group hover:bg-[#151f33] transition-all duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-secondary-container/10 flex items-center justify-center mb-6 group-hover:bg-secondary-container/20 transition-colors">
                <Bot className="text-secondary w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold syne mb-4 text-on-surface">AI Detection</h3>
              <p className="text-on-surface-variant leading-relaxed text-sm">Proprietary deep-learning models detect subtle lesions and fat levels that might be missed by the naked eye.</p>
            </motion.div>
            
            {/* Feature 3 */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="p-8 rounded-2xl bg-[#0D1424] border border-white/5 group hover:bg-[#151f33] transition-all duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-tertiary/10 flex items-center justify-center mb-6 group-hover:bg-tertiary/20 transition-colors">
                <LineChart className="text-tertiary w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold syne mb-4 text-on-surface">Track Over Time</h3>
              <p className="text-on-surface-variant leading-relaxed text-sm">Compare historical scans to monitor progression or recovery with automated side-by-side visualization.</p>
            </motion.div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-12 px-6 relative z-20 bg-[#070B14]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-2 opacity-50">
            <div className="w-6 h-6 rounded bg-on-surface-variant flex items-center justify-center">
              <Microscope className="text-[#070B14] w-4 h-4" />
            </div>
            <span className="syne font-bold text-on-surface tracking-tighter">ScanSight</span>
          </div>
          
          <div className="flex flex-wrap justify-center gap-8 text-sm text-outline">
            <a className="hover:text-primary transition-colors" href="#">Privacy Policy</a>
            <a className="hover:text-primary transition-colors" href="#">Clinical Ethics</a>
            <a className="hover:text-primary transition-colors" href="#">API Docs</a>
            <a className="hover:text-primary transition-colors" href="#">Support</a>
          </div>
          
          <div className="text-outline text-xs font-mono tracking-wider">
            © 2024 SCANSIGHT SYSTEMS INC.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
