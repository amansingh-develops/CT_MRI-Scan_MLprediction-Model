import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useUiStore } from '../stores/uiStore';
import { Settings as SettingsIcon, Edit, Lock, Shield, ChevronRight, AlertTriangle, Trash2, Moon, Sun, Save, X, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../lib/utils';
import { auth, db } from '../lib/firebase';
import { updatePassword, EmailAuthProvider, reauthenticateWithCredential, deleteUser, signOut } from 'firebase/auth';
import { doc, updateDoc, deleteDoc } from 'firebase/firestore';

const Settings = () => {
  const { user, setUser, clearUser } = useAuthStore();
  const { theme, toggleTheme } = useUiStore();
  const navigate = useNavigate();

  // Profile editing state
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileForm, setProfileForm] = useState({
    name: user?.name || '',
    phone: user?.phone || '',
    medicalId: user?.medicalId || '',
  });
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileMessage, setProfileMessage] = useState(null);

  // Password change state
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [passwordForm, setPasswordForm] = useState({ currentPassword: '', newPassword: '', confirmPassword: '' });
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [passwordError, setPasswordError] = useState(null);
  const [passwordSuccess, setPasswordSuccess] = useState(false);

  // Delete account state
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [deletePassword, setDeletePassword] = useState('');
  const [deleteError, setDeleteError] = useState(null);

  // Notification preference
  const [emailNotifications, setEmailNotifications] = useState(true);

  // === Profile Update ===
  const handleProfileSave = async () => {
    setProfileSaving(true);
    setProfileMessage(null);
    try {
      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        name: profileForm.name,
        ...(user.role === 'patient' && { phone: profileForm.phone }),
        ...(user.role === 'clinician' && { medicalId: profileForm.medicalId }),
      });
      setUser({ ...user, name: profileForm.name, phone: profileForm.phone, medicalId: profileForm.medicalId });
      setProfileMessage('Profile updated successfully!');
      setIsEditingProfile(false);
    } catch (err) {
      setProfileMessage('Failed to update: ' + err.message);
    } finally {
      setProfileSaving(false);
    }
  };

  // === Password Change ===
  const handlePasswordChange = async () => {
    setPasswordError(null);
    setPasswordSuccess(false);
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      setPasswordError('Passwords do not match.');
      return;
    }
    if (passwordForm.newPassword.length < 6) {
      setPasswordError('New password must be at least 6 characters.');
      return;
    }
    setPasswordLoading(true);
    try {
      const credential = EmailAuthProvider.credential(auth.currentUser.email, passwordForm.currentPassword);
      await reauthenticateWithCredential(auth.currentUser, credential);
      await updatePassword(auth.currentUser, passwordForm.newPassword);
      setPasswordSuccess(true);
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
      setTimeout(() => setShowPasswordModal(false), 1500);
    } catch (err) {
      if (err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
        setPasswordError('Current password is incorrect.');
      } else {
        setPasswordError(err.message);
      }
    } finally {
      setPasswordLoading(false);
    }
  };

  // === Delete Account ===
  const handleDeleteAccount = async () => {
    setDeleteError(null);
    setDeleteLoading(true);
    try {
      // Re-authenticate if email/password user
      if (auth.currentUser.providerData[0]?.providerId === 'password') {
        const credential = EmailAuthProvider.credential(auth.currentUser.email, deletePassword);
        await reauthenticateWithCredential(auth.currentUser, credential);
      }
      // Delete Firestore user document
      await deleteDoc(doc(db, 'users', user.uid));
      // Delete Firebase Auth account
      await deleteUser(auth.currentUser);
      clearUser();
      navigate('/');
    } catch (err) {
      if (err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
        setDeleteError('Incorrect password.');
      } else {
        setDeleteError(err.message);
      }
    } finally {
      setDeleteLoading(false);
    }
  };

  // === Notification Toggle ===
  const handleNotificationToggle = async () => {
    const newValue = !emailNotifications;
    setEmailNotifications(newValue);
    try {
      await updateDoc(doc(db, 'users', user.uid), { emailNotifications: newValue });
    } catch (err) {
      console.warn('Could not persist notification preference:', err.message);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto min-h-screen bg-background">
      <div className="max-w-5xl mx-auto p-6 lg:p-12">
        <header className="mb-10">
          <h1 className="syne text-4xl font-bold tracking-tight text-on-surface mb-2">User Settings</h1>
          <p className="text-on-surface-variant font-body">Manage your professional profile and application preferences.</p>
        </header>

        {/* Success/Error Message */}
        <AnimatePresence>
          {profileMessage && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mb-6 p-4 rounded-lg bg-primary/10 text-primary text-sm font-medium">
              {profileMessage}
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-12 gap-6">
          
          {/* Profile Section */}
          <section className="col-span-12 lg:col-span-8 bg-surface-container rounded-xl p-8 flex flex-col md:flex-row gap-8 items-start relative overflow-hidden border border-outline-variant/10">
            <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-[80px] -mr-32 -mt-32"></div>
            
            <div className="relative group shrink-0">
              <img 
                alt="Profile" 
                className="w-32 h-32 md:w-40 md:h-40 rounded-xl object-cover border-2 border-outline-variant/30" 
                src={
                  user?.role === 'clinician' 
                    ? "https://lh3.googleusercontent.com/aida-public/AB6AXuBi8-b3qQO0kcGc4JJQT0a4T4BEHy0EMiyMwS3qtkwbkc3E6poFruI09KadfN2uEEuGP-JCaCOBy-EcN6Fbf6NCMTg7-cNn5TGNEvZm8-KFcqIJaHtf9SU7JwFOwOYeWj0SP4-NVH31-0CBb9HGvedz_66KYZRQFr-SL-lYww3apnsf2HSD8X_qnfL6pRvVyXztg97W9Vypd1RkIFkjRLYr6p9mVmi9Yr-KPa9AwfN85MYlyGFucUIk39CRHGGf5BihyeIhdjFOrEg"
                    : "https://lh3.googleusercontent.com/aida-public/AB6AXuB3aJ6TmcIVCzP7J9fcr4Dg0gQaEf840lLOga2eDnM9T1c8Vqc34V6g4rGF8i9XWEk7QI_Iq2ltRnuVKlYIJWjO3dL8aEHzTdwFRTHiALkYviHqPLLVhA7fVN5RXysPAPIScyqApyTX8vPVeeLg4McfL0T3teVlpEdFUumDE_-1D5rxVr2o8jKu_ZixOa2reRs4PrJHu0vpGQwHr0EpnHZfvOuqOiHxD-ag7Q6hpoAvtBTvgcdIBbEXBIp7Ps2808yYSTF5AG7lOX0"
                }
              />
              <button 
                onClick={() => setIsEditingProfile(!isEditingProfile)} 
                className="absolute bottom-2 right-2 bg-primary text-on-primary p-2 rounded-lg shadow-lg hover:bg-primary-container transition-colors"
              >
                {isEditingProfile ? <X className="w-4 h-4" /> : <Edit className="w-4 h-4" />}
              </button>
            </div>
            
            <div className="flex-grow">
              <div className="flex flex-wrap items-center gap-3 mb-2">
                {isEditingProfile ? (
                  <input 
                    value={profileForm.name}
                    onChange={(e) => setProfileForm({ ...profileForm, name: e.target.value })}
                    className="text-2xl font-bold syne text-on-surface bg-surface-container-lowest rounded-lg px-3 py-1 border border-outline-variant/30 focus:ring-2 focus:ring-primary/30 outline-none"
                  />
                ) : (
                  <h2 className="text-2xl font-bold syne text-on-surface">{user?.name || 'User'}</h2>
                )}
                <span className="bg-secondary-container/10 text-secondary border border-secondary/20 px-3 py-0.5 rounded-full text-[10px] font-bold tracking-widest uppercase">
                  {user?.role === 'clinician' ? 'Clinician' : 'Patient'}
                </span>
              </div>
              <p className="text-on-surface-variant font-body mb-6">{user?.email || 'user@scansight.ai'}</p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {user?.role === 'clinician' ? (
                  <div className="space-y-1">
                    <label className="text-[10px] uppercase font-bold tracking-widest text-on-surface-variant">NMC ID</label>
                    {isEditingProfile ? (
                      <input
                        value={profileForm.medicalId}
                        onChange={(e) => setProfileForm({ ...profileForm, medicalId: e.target.value })}
                        className="font-mono text-sm text-primary bg-surface-container-lowest rounded-lg px-3 py-1.5 border border-outline-variant/30 focus:ring-2 focus:ring-primary/30 outline-none w-full"
                      />
                    ) : (
                      <p className="font-mono text-sm text-primary">{user?.medicalId || 'Not set'}</p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-1">
                    <label className="text-[10px] uppercase font-bold tracking-widest text-on-surface-variant">Phone</label>
                    {isEditingProfile ? (
                      <input
                        value={profileForm.phone}
                        onChange={(e) => setProfileForm({ ...profileForm, phone: e.target.value })}
                        className="font-mono text-sm text-primary bg-surface-container-lowest rounded-lg px-3 py-1.5 border border-outline-variant/30 focus:ring-2 focus:ring-primary/30 outline-none w-full"
                      />
                    ) : (
                      <p className="font-mono text-sm text-primary">{user?.phone || 'Not set'}</p>
                    )}
                  </div>
                )}
                <div className="space-y-1">
                  <label className="text-[10px] uppercase font-bold tracking-widest text-on-surface-variant">Total Scans</label>
                  <p className="font-body text-sm text-on-surface">{user?.totalScans ?? 0}</p>
                </div>
              </div>

              {/* Save button when editing */}
              {isEditingProfile && (
                <button
                  onClick={handleProfileSave}
                  disabled={profileSaving}
                  className="mt-6 px-6 py-2 bg-primary text-on-primary rounded-lg font-bold text-sm flex items-center gap-2 hover:brightness-110 active:scale-95 transition-all disabled:opacity-50"
                >
                  {profileSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                  {profileSaving ? 'Saving...' : 'Save Changes'}
                </button>
              )}
            </div>
          </section>

          {/* Stats/About Summary */}
          <section className="col-span-12 lg:col-span-4 bg-surface-container-high rounded-xl p-8 flex flex-col justify-between border border-outline-variant/10 shadow-sm">
            <div>
              <h3 className="text-sm font-bold tracking-widest uppercase text-on-surface-variant mb-4">About ScanSight</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-on-surface">App Version</span>
                  <span className="font-mono text-xs bg-surface-container px-2 py-1 rounded">v2.4.0-stable</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-on-surface">Model Engine</span>
                  <span className="font-mono text-xs bg-surface-container px-2 py-1 rounded">FasNet-V3</span>
                </div>
              </div>
            </div>
            <div className="mt-8">
              <p className="text-xs text-on-surface-variant leading-relaxed">
                Developed in partnership with FasNet Research Labs. All AI analysis is verified against clinical standards.
              </p>
            </div>
          </section>

          {/* Preferences */}
          <section className="col-span-12 md:col-span-6 bg-surface-container rounded-xl p-8 border border-outline-variant/5">
            <h3 className="text-lg font-bold syne mb-6 flex items-center gap-2 text-on-surface">
              <SettingsIcon className="w-5 h-5 text-primary" />
              Preferences
            </h3>
            <div className="space-y-6">
              
              <div className="flex items-center justify-between p-4 bg-surface-container-low rounded-lg border border-outline-variant/5">
                <div>
                  <p className="text-sm font-bold text-on-surface">Interface Theme</p>
                  <p className="text-xs text-on-surface-variant">Switch between Dark and Light mode</p>
                </div>
                <div className="flex bg-surface-container-highest p-1 rounded-full border border-outline-variant/10">
                  <button 
                    onClick={() => theme !== 'dark' && toggleTheme()}
                    className={cn("p-2 rounded-full transition-colors", theme === 'dark' ? "bg-primary text-on-primary shadow-sm" : "text-on-surface-variant hover:text-on-surface")}
                  >
                    <Moon className="w-4 h-4" />
                  </button>
                  <button 
                    onClick={() => theme !== 'light' && toggleTheme()}
                    className={cn("p-2 rounded-full transition-colors", theme === 'light' ? "bg-primary text-on-primary shadow-sm" : "text-on-surface-variant hover:text-on-surface")}
                  >
                    <Sun className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between p-4 bg-surface-container-low rounded-lg border border-outline-variant/5">
                <div>
                  <p className="text-sm font-bold text-on-surface">Email Notifications</p>
                  <p className="text-xs text-on-surface-variant">Receive scan analysis alerts via email</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    checked={emailNotifications} 
                    onChange={handleNotificationToggle}
                    className="sr-only peer" 
                    type="checkbox" 
                  />
                  <div className="w-11 h-6 bg-surface-container-highest peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-container"></div>
                </label>
              </div>

            </div>
          </section>

          {/* Account Security */}
          <section className="col-span-12 md:col-span-6 bg-surface-container rounded-xl p-8 border border-outline-variant/5">
            <h3 className="text-lg font-bold syne mb-6 flex items-center gap-2 text-on-surface">
              <Shield className="w-5 h-5 text-primary" />
              Account Security
            </h3>
            <div className="space-y-4">
              <button 
                onClick={() => setShowPasswordModal(true)}
                className="w-full flex items-center justify-between p-4 bg-surface-container-low rounded-lg hover:bg-surface-container-high transition-colors group border border-outline-variant/5"
              >
                <div className="flex items-center gap-3 text-left">
                  <Lock className="w-5 h-5 text-on-surface-variant group-hover:text-primary transition-colors" />
                  <div>
                    <p className="text-sm font-bold text-on-surface">Change Password</p>
                    <p className="text-xs text-on-surface-variant">Update your login credentials</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5 text-on-surface-variant" />
              </button>
              
              <button className="w-full flex items-center justify-between p-4 bg-surface-container-low rounded-lg hover:bg-surface-container-high transition-colors group border border-outline-variant/5">
                <div className="flex items-center gap-3 text-left">
                  <Shield className="w-5 h-5 text-on-surface-variant group-hover:text-primary transition-colors" />
                  <div>
                    <p className="text-sm font-bold text-on-surface">Two-Factor Auth</p>
                    <p className="text-xs text-on-surface-variant">Coming soon</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5 text-on-surface-variant" />
              </button>
            </div>
          </section>

          {/* Danger Zone */}
          <section className="col-span-12 border-2 border-error/20 bg-error-container/5 rounded-xl p-8 shadow-sm">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
              <div>
                <h3 className="text-lg font-bold syne text-error flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Danger Zone
                </h3>
                <p className="text-sm text-on-surface-variant mt-1">Permanently delete your account and all associated scan data. This action is irreversible.</p>
              </div>
              <button 
                onClick={() => setShowDeleteConfirm(true)}
                className="px-6 py-3 bg-error text-on-error font-bold rounded-lg hover:bg-error/90 transition-all flex items-center gap-2 shrink-0 shadow-[0_0_15px_rgba(255,180,171,0.2)]"
              >
                <Trash2 className="w-4 h-4" />
                Delete Account
              </button>
            </div>
          </section>

        </div>
      </div>

      {/* === Password Change Modal === */}
      <AnimatePresence>
        {showPasswordModal && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setShowPasswordModal(false)}
          >
            <motion.div 
              initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-surface-container rounded-xl p-8 w-full max-w-md border border-outline-variant/20"
            >
              <h3 className="syne text-xl font-bold text-on-surface mb-6">Change Password</h3>
              <div className="space-y-4">
                <input
                  type="password" placeholder="Current Password"
                  value={passwordForm.currentPassword}
                  onChange={(e) => setPasswordForm({ ...passwordForm, currentPassword: e.target.value })}
                  className="w-full h-12 px-4 bg-surface-container-lowest rounded-lg text-on-surface placeholder:text-on-surface-variant/40 border-none outline-none focus:ring-2 focus:ring-primary/30"
                />
                <input
                  type="password" placeholder="New Password"
                  value={passwordForm.newPassword}
                  onChange={(e) => setPasswordForm({ ...passwordForm, newPassword: e.target.value })}
                  className="w-full h-12 px-4 bg-surface-container-lowest rounded-lg text-on-surface placeholder:text-on-surface-variant/40 border-none outline-none focus:ring-2 focus:ring-primary/30"
                />
                <input
                  type="password" placeholder="Confirm New Password"
                  value={passwordForm.confirmPassword}
                  onChange={(e) => setPasswordForm({ ...passwordForm, confirmPassword: e.target.value })}
                  className="w-full h-12 px-4 bg-surface-container-lowest rounded-lg text-on-surface placeholder:text-on-surface-variant/40 border-none outline-none focus:ring-2 focus:ring-primary/30"
                />
                {passwordError && <p className="text-error text-sm">{passwordError}</p>}
                {passwordSuccess && <p className="text-emerald-400 text-sm">Password updated successfully!</p>}
              </div>
              <div className="flex gap-3 mt-6">
                <button onClick={() => setShowPasswordModal(false)} className="flex-1 py-3 rounded-lg bg-surface-container-highest text-on-surface font-bold text-sm">Cancel</button>
                <button 
                  onClick={handlePasswordChange} disabled={passwordLoading}
                  className="flex-1 py-3 rounded-lg bg-primary text-on-primary font-bold text-sm flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {passwordLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Update Password'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* === Delete Account Confirmation Modal === */}
      <AnimatePresence>
        {showDeleteConfirm && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setShowDeleteConfirm(false)}
          >
            <motion.div 
              initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-surface-container rounded-xl p-8 w-full max-w-md border-2 border-error/30"
            >
              <h3 className="syne text-xl font-bold text-error mb-2 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                Delete Account
              </h3>
              <p className="text-on-surface-variant text-sm mb-6">This will permanently delete your account, all scan data, and chat history. This cannot be undone.</p>
              {auth.currentUser?.providerData[0]?.providerId === 'password' && (
                <input
                  type="password" placeholder="Enter your password to confirm"
                  value={deletePassword}
                  onChange={(e) => setDeletePassword(e.target.value)}
                  className="w-full h-12 px-4 bg-surface-container-lowest rounded-lg text-on-surface placeholder:text-on-surface-variant/40 border-none outline-none focus:ring-2 focus:ring-error/30 mb-4"
                />
              )}
              {deleteError && <p className="text-error text-sm mb-4">{deleteError}</p>}
              <div className="flex gap-3">
                <button onClick={() => setShowDeleteConfirm(false)} className="flex-1 py-3 rounded-lg bg-surface-container-highest text-on-surface font-bold text-sm">Cancel</button>
                <button 
                  onClick={handleDeleteAccount} disabled={deleteLoading}
                  className="flex-1 py-3 rounded-lg bg-error text-on-error font-bold text-sm flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {deleteLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Delete Forever'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Settings;
