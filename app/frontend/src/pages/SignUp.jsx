import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Stethoscope, ArrowRight, CheckCircle, Eye, EyeOff, Calendar, Phone } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuthStore } from '../stores/authStore';
import { createUserWithEmailAndPassword, updateProfile, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { auth, db } from '../lib/firebase';
import { z } from 'zod';

const patientSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Invalid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
  phone: z.string().regex(/^[0-9]{10}$/, "Phone number must be exactly 10 digits"),
  dob: z.string().min(1, "Date of birth is required"),
});

const clinicianSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Invalid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
  medicalId: z.string().regex(/^[A-Z]{2}-[0-9]{4}-[A-Z]{2}$/, "NMC ID must follow format XX-0000-XX"),
});

const SignUp = () => {
  const navigate = useNavigate();
  const { setUser } = useAuthStore();
  const [accountType, setAccountType] = useState('patient'); // 'patient' or 'clinician'
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    medicalId: '',
    dob: '',
    phone: '',
    email: '',
    password: ''
  });

  const handleSignUp = async (e) => {
    e.preventDefault();
    setError(null);

    // Validation
    try {
      if (accountType === 'patient') {
        patientSchema.parse(formData);
      } else {
        clinicianSchema.parse(formData);
      }
    } catch (err) {
      if (err instanceof z.ZodError) {
        setError(err.errors[0].message);
        return;
      }
    }

    setLoading(true);

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      const displayName = formData.name || (accountType === 'clinician' ? 'Dr. Ankit Verma' : 'Priya Sharma');
      
      await updateProfile(userCredential.user, { displayName });

      // Save user profile to Firestore with role-specific fields
      const userProfile = {
        uid: userCredential.user.uid,
        name: displayName,
        email: userCredential.user.email,
        role: accountType,
        createdAt: serverTimestamp(),
        totalScans: 0,
        ...(accountType === 'patient' && {
          phone: formData.phone,
          dob: formData.dob,
        }),
        ...(accountType === 'clinician' && {
          medicalId: formData.medicalId,
        }),
      };
      await setDoc(doc(db, 'users', userCredential.user.uid), userProfile);

      setUser(userProfile);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Failed to create account.');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    setError(null);
    setLoading(true);
    const provider = new GoogleAuthProvider();
    try {
      const userCredential = await signInWithPopup(auth, provider);

      // Save user profile to Firestore (Google provides name + email automatically)
      const userProfile = {
        uid: userCredential.user.uid,
        name: userCredential.user.displayName,
        email: userCredential.user.email,
        role: accountType,
        createdAt: serverTimestamp(),
        totalScans: 0,
      };
      await setDoc(doc(db, 'users', userCredential.user.uid), userProfile, { merge: true });

      setUser(userProfile);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Failed to sign up with Google.');
    } finally {
      setLoading(false);
    }
  };

  // Input style constants
  const inputClass = "w-full h-12 px-4 bg-surface-container-lowest border-none rounded-lg text-on-surface placeholder:text-surface-variant focus:ring-2 focus:ring-primary/30 outline-none transition-all";

  return (
    <div className="min-h-screen flex w-full overflow-hidden">
      {/* Left Section: 60% Form Canvas */}
      <main className="w-full lg:w-[60%] flex flex-col bg-background relative overflow-y-auto px-8 py-12 md:px-20 md:py-16">
        {/* Top App Bar Mockup */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-between items-center w-full mb-16"
        >
          <div className="flex items-center gap-2">
            <span className="text-2xl font-black text-primary tracking-tighter syne">ScanSight</span>
          </div>
          <Link className="text-sm font-medium text-on-surface-variant hover:text-primary transition-colors" to="/signin">
            Already have an account? <span className="text-primary underline underline-offset-4 ml-1">Sign In</span>
          </Link>
        </motion.div>

        <div className="max-w-2xl w-full mx-auto">
          <motion.header 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-10"
          >
            <h1 className="text-5xl syne font-bold text-on-surface tracking-tight mb-4">Join ScanSight</h1>
            <p className="text-on-surface-variant text-lg">
              {accountType === 'patient' 
                ? "Create your account in under a minute — it's quick and easy."
                : "Select your account type to begin the clinical onboarding process."}
            </p>
          </motion.header>

          {/* Role Selection Grid */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12"
          >
            {/* Patient Card */}
            <button 
              type="button"
              onClick={() => setAccountType('patient')}
              className={cn(
                "group relative flex flex-col items-start p-6 bg-surface-container rounded-xl transition-all duration-300 text-left outline-none",
                accountType === 'patient' 
                  ? "border-2 border-primary ring-4 ring-primary/10" 
                  : "border-2 border-transparent hover:border-primary/30 focus:ring-2 focus:ring-primary/20"
              )}
            >
              <div className={cn(
                "w-12 h-12 rounded-lg flex items-center justify-center mb-6 transition-colors",
                accountType === 'patient' ? "bg-primary-container" : "bg-surface-container-high group-hover:bg-primary-container"
              )}>
                <User className={cn("w-6 h-6", accountType === 'patient' ? "text-on-primary" : "text-primary group-hover:text-on-primary")} />
              </div>
              <h3 className="text-xl font-bold text-on-surface mb-2">Patient</h3>
              <p className="text-sm text-on-surface-variant leading-relaxed">Securely access your imaging results and connect with your care team.</p>
              <div className={cn(
                "absolute top-4 right-4 transition-opacity",
                accountType === 'patient' ? "opacity-100" : "opacity-0 group-hover:opacity-100"
              )}>
                <CheckCircle className="w-6 h-6 text-primary fill-primary/20" />
              </div>
            </button>

            {/* Doctor Card */}
            <button 
              type="button"
              onClick={() => setAccountType('clinician')}
              className={cn(
                "group relative flex flex-col items-start p-6 bg-surface-container rounded-xl transition-all duration-300 text-left outline-none",
                accountType === 'clinician' 
                  ? "border-2 border-primary ring-4 ring-primary/10" 
                  : "border-2 border-transparent hover:border-primary/30 focus:ring-2 focus:ring-primary/20"
              )}
            >
              <div className={cn(
                "w-12 h-12 rounded-lg flex items-center justify-center mb-6 transition-colors",
                accountType === 'clinician' ? "bg-primary-container" : "bg-surface-container-high group-hover:bg-primary-container"
              )}>
                <Stethoscope className={cn("w-6 h-6", accountType === 'clinician' ? "text-on-primary" : "text-primary group-hover:text-on-primary")} />
              </div>
              <h3 className="text-xl font-bold text-on-surface mb-2">Clinician</h3>
              <p className="text-sm text-on-surface-variant leading-relaxed">Professional tools for AI-assisted diagnostics and radiology management.</p>
              <div className={cn(
                "absolute top-4 right-4 transition-opacity",
                accountType === 'clinician' ? "opacity-100" : "opacity-0 group-hover:opacity-100"
              )}>
                <CheckCircle className="w-6 h-6 text-primary fill-primary/20" />
              </div>
            </button>
          </motion.div>

          {/* Signup Form — adapts based on accountType */}
          <motion.form 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            onSubmit={handleSignUp} 
            className="space-y-6"
          >
            <AnimatePresence mode="wait">
              <motion.div 
                key={accountType}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                {accountType === 'patient' ? (
                  <>
                    {/* ── Patient Fields ── */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">Full Name</label>
                        <input 
                          className={inputClass}
                          placeholder="Priya Sharma" 
                          type="text"
                          value={formData.name}
                          onChange={(e) => setFormData({...formData, name: e.target.value})}
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">Date of Birth</label>
                        <input 
                          className={inputClass}
                          placeholder="DD / MM / YYYY" 
                          type="text"
                          value={formData.dob}
                          onChange={(e) => setFormData({...formData, dob: e.target.value})}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">Phone Number</label>
                        <input 
                          className={inputClass}
                          placeholder="+91 98765 43210" 
                          type="tel"
                          value={formData.phone}
                          onChange={(e) => setFormData({...formData, phone: e.target.value})}
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">Email</label>
                        <input 
                          className={inputClass}
                          placeholder="priya.sharma@gmail.com" 
                          type="email"
                          value={formData.email}
                          onChange={(e) => setFormData({...formData, email: e.target.value})}
                        />
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    {/* ── Clinician Fields ── */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">Full Name</label>
                        <input 
                          className={inputClass}
                          placeholder="Dr. Ankit Verma" 
                          type="text"
                          value={formData.name}
                          onChange={(e) => setFormData({...formData, name: e.target.value})}
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">NMC Registration No.</label>
                        <input 
                          className={cn(inputClass, "font-mono")}
                          placeholder="NMC-2024-123456" 
                          type="text"
                          value={formData.medicalId}
                          onChange={(e) => setFormData({...formData, medicalId: e.target.value})}
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">Institutional Email</label>
                      <input 
                        className={inputClass}
                        placeholder="ankit.verma@aiims.edu.in" 
                        type="email"
                        value={formData.email}
                        onChange={(e) => setFormData({...formData, email: e.target.value})}
                      />
                    </div>
                  </>
                )}

                {/* Password — shared by both roles */}
                <div className="space-y-2">
                  <label className="block text-[0.7rem] uppercase tracking-[0.05em] font-bold text-on-surface-variant px-1">Password</label>
                  <div className="relative">
                    <input 
                      className={inputClass}
                      placeholder="••••••••••••" 
                      type={showPassword ? "text" : "password"}
                      value={formData.password}
                      onChange={(e) => setFormData({...formData, password: e.target.value})}
                    />
                    <button 
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-1/2 -translate-y-1/2 text-on-surface-variant hover:text-on-surface transition-colors" 
                      type="button"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>

            <div className="pt-6">
              {error && (
                <div className="mb-4 p-3 rounded-lg bg-error-container/20 border border-error/50 text-error text-sm">
                  {error}
                </div>
              )}
              <button 
                type="submit"
                disabled={loading}
                className="w-full h-14 rounded-xl flex items-center justify-center gap-3 text-on-primary font-bold text-sm hover:opacity-90 active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ background: 'linear-gradient(to bottom, #89ceff, #0ea5e9)', boxShadow: 'inset 0 1px 0 0 rgba(255, 255, 255, 0.2)' }}
              >
                {loading ? 'Creating Account...' : (accountType === 'patient' ? 'Get Started' : 'Create Professional Account')}
                {!loading && <ArrowRight className="w-5 h-5" />}
              </button>

              {/* Divider */}
              <div className="relative flex py-2 items-center mt-4">
                <div className="flex-grow border-t border-outline-variant/30"></div>
                <span className="shrink-0 px-4 text-outline text-xs">OR</span>
                <div className="flex-grow border-t border-outline-variant/30"></div>
              </div>

              {/* Google Sign In Button */}
              <button 
                type="button"
                onClick={handleGoogleSignUp}
                disabled={loading}
                className="w-full h-14 mt-4 bg-surface-container-low border border-outline-variant/30 hover:bg-surface-container transition-colors rounded-xl flex items-center justify-center gap-3 text-on-surface font-bold text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <svg viewBox="0 0 24 24" className="w-5 h-5">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                </svg>
                Sign up with Google
              </button>
            </div>
            
            <p className="text-center text-xs text-on-surface-variant px-4">
              By proceeding, you agree to our <a className="text-primary hover:underline" href="#">Terms of Service</a> and <a className="text-primary hover:underline" href="#">Privacy Policy</a>.
            </p>
          </motion.form>
        </div>
      </main>

      {/* Right Section: 40% Visual Anchor */}
      <aside className="hidden lg:flex lg:w-[40%] bg-surface-container-low flex-col items-center justify-center p-12 relative overflow-hidden">
        {/* Abstract Background Glows */}
        <div className="absolute top-1/4 -right-20 w-96 h-96 bg-primary/5 blur-[120px] rounded-full"></div>
        <div className="absolute bottom-1/4 -left-20 w-96 h-96 bg-secondary/5 blur-[120px] rounded-full"></div>
        
        {/* Medical CT Scan Illustration Container */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          className="relative w-full max-w-md aspect-square bg-background rounded-2xl p-8 border border-outline-variant/10 shadow-2xl flex flex-col items-center justify-center group"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-50 rounded-2xl"></div>
          
          {/* Scan Simulator Graphic */}
          <div className="relative w-full h-full border border-primary/20 rounded-lg overflow-hidden flex items-center justify-center">
            <img 
              alt="Medical Imaging Brain Scan" 
              className="w-full h-full object-cover grayscale opacity-40 mix-blend-screen" 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuCRcyRQUGYXrgIoeGu0NnzuZ9EAQ9P7TvpJzpcQ2QuPKovEW7diOTN4RSHU2Cefsp5grq6-RunNHIlKPq4YGot-DDdl2BkiNNEw4fKAQj-3jq8fBvWjy6xYceNbOeXPelIAsG1K_EKQOnG9H9NPM7nDOS98wgGZdSLwIcP8uVQ5pfivQ0gI-16vaTEsME9FTB_cBwrg2nVpz42jbqm0d1ARKjLeAOUDDwR_CiLECTVJk_6_bt0K5tleV3icDftZHY07xsU-_2eJ3VA"
            />
            
            {/* Scanning Line Animation */}
            <motion.div 
              animate={{ top: ['0%', '100%'], opacity: [0, 1, 1, 0] }}
              transition={{ duration: 4, ease: "linear", repeat: Infinity }}
              className="absolute left-0 w-full h-1 bg-primary/40 shadow-[0_0_15px_rgba(137,206,255,0.6)]"
            ></motion.div>
            
            {/* UI HUD Overlays */}
            <div className="absolute top-4 left-4 flex flex-col gap-1">
              <span className="text-[10px] font-mono text-primary uppercase font-bold tracking-widest">Live Analysis</span>
              <div className="flex gap-1">
                <div className="w-1 h-1 bg-primary rounded-full"></div>
                <div className="w-1 h-1 bg-primary rounded-full animate-pulse"></div>
                <div className="w-1 h-1 bg-primary rounded-full"></div>
              </div>
            </div>
            
            <div className="absolute bottom-4 right-4 text-[10px] font-mono text-on-surface-variant text-right">
              RES: 2048 x 2048px<br/>DEPTH: 16-BIT
            </div>
          </div>
        </motion.div>

        {/* Quote */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-16 text-center max-w-sm"
        >
          <blockquote className="syne text-2xl font-bold text-on-surface leading-tight mb-4">
            "Early detection saves lives."
          </blockquote>
          <p className="text-on-surface-variant text-sm font-medium">
            Harnessing the power of precision diagnostics to transform patient outcomes globally.
          </p>
        </motion.div>

        {/* Decorative Elements */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-6 w-max"
        >
          <div className="flex -space-x-3">
            <div className="w-8 h-8 rounded-full border-2 border-surface-container-low bg-surface-container-high overflow-hidden">
              <img alt="Doctor 1" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBibuTD9fZdLFfSdBrchANWT4EuXJVpxYAxumtO95S5R74m7ziN1SqtjRXL5QxjwgTcA2uhUFZ3tiyUfBaB6g9LRuIQDYKa7iuYiGHl4-E6Bb17c_w2ylW6e09mDZYN2B7vP99KYl4KPSYalkzPNcYClsWJy2cr8rSDLNLRuc0gHw4cDk-ubb5PlhHYRJwQHtoH_0a2luBguLZbq4LFtykoMx3CjoAI5nfjcjmqOYUcRD2LTvdEn3Ez_LsxyOy-x3s1rKTnJ_6MNPw"/>
            </div>
            <div className="w-8 h-8 rounded-full border-2 border-surface-container-low bg-surface-container-high overflow-hidden">
              <img alt="Doctor 2" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBLXu37jmLS9hGit1KQFYeqv-9qyC9WANouUJ6ZtfjHLCj5YcrzXFQ8kbPizn1zGpsp2o8-qrgDwUee6Uwz-o6oMGEA5gOILHf6TozxMKBDmbzjKWFWFLYNpOFGAewZ47IojKTgO6TdyHXxy-Xpq4xbz61oYgIK1q0CZQH4wUPyUQpF12lXNDwpIUFx3d6VgQJigF8B5RP3vXxzZ4Cy4anfpeKm5wj4LMu8rEwRI4wDeWxSRA2ZVJHu07oEarh_wb6qZ9NYcPeO9ZA"/>
            </div>
            <div className="w-8 h-8 rounded-full border-2 border-surface-container-low bg-surface-container-high overflow-hidden">
              <img alt="Doctor 3" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCiZj9Lk1X8FfGQv8QQhRo7Gzc0GCJeWpg8RPrauqcgmJZi5Isd2Tjv2dhu3VD0ttt-M24Tvc_DrJDO2tlaYvYO8BMxuxDaj6EfYsDv_VfWieskN3qsLcMcirKC8YpWdgVf_0nLOkuxSPJ4DcX8pGL2AolRj3ws-XrF4-RnZEbCanuNF0PnK_m8PeGDSLQ4C6M_YJjyNea8fmfSnMhzVWuQ-qCNQ60X1tADgng1RlX92v3QVccRLVz1vAmwnCQUl8r-h8nVLDT5Trg"/>
            </div>
          </div>
          <span className="text-xs text-on-surface-variant font-bold tracking-wide uppercase whitespace-nowrap">Trusted by 2k+ doctors across India</span>
        </motion.div>
      </aside>
    </div>
  );
};

export default SignUp;
