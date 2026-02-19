// ============================================
// AI HR MANAGER - COMPLETE BACKEND
// Node.js + Express + MongoDB + AI APIs
// ============================================

// Package.json dependencies needed:
/*
{
  "name": "ai-hr-manager-backend",
  "version": "1.0.0",
  "description": "AI-powered HR management backend",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "mongoose": "^7.0.0",
    "dotenv": "^16.0.3",
    "cors": "^2.8.5",
    "bcryptjs": "^2.4.3",
    "jsonwebtoken": "^9.0.0",
    "multer": "^1.4.5-lts.1",
    "pdf-parse": "^1.1.1",
    "natural": "^6.0.0",
    "node-cron": "^3.0.2",
    "nodemailer": "^6.9.1",
    "openai": "^4.0.0",
    "axios": "^1.4.0"
  }
}
*/

// ============================================
// SERVER.JS - Main Application
// ============================================

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');
const cron = require('node-cron');

dotenv.config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/ai-hr-manager', {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => console.log('✅ MongoDB Connected'))
  .catch(err => console.error('❌ MongoDB Error:', err));

// Import Routes
const authRoutes = require('./routes/auth');
const employeeRoutes = require('./routes/employees');
const recruitmentRoutes = require('./routes/recruitment');
const performanceRoutes = require('./routes/performance');
const chatbotRoutes = require('./routes/chatbot');
const retentionRoutes = require('./routes/retention');
const sentimentRoutes = require('./routes/sentiment');

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/employees', employeeRoutes);
app.use('/api/recruitment', recruitmentRoutes);
app.use('/api/performance', performanceRoutes);
app.use('/api/chatbot', chatbotRoutes);
app.use('/api/retention', retentionRoutes);
app.use('/api/sentiment', sentimentRoutes);

// Health Check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'AI HR Manager API is running' });
});

// Scheduled Jobs
// Run retention risk analysis daily at 2 AM
cron.schedule('0 2 * * *', async () => {
  console.log('🔄 Running daily retention risk analysis...');
  const RetentionService = require('./services/retentionService');
  await RetentionService.analyzeBatch();
});

// Run sentiment analysis every 6 hours
cron.schedule('0 */6 * * *', async () => {
  console.log('🔄 Running sentiment analysis...');
  const SentimentService = require('./services/sentimentService');
  await SentimentService.analyzeBatch();
});

// Error Handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});

// ============================================
// MODELS/Employee.js
// ============================================

const employeeSchema = new mongoose.Schema({
  firstName: { type: String, required: true },
  lastName: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  department: { type: String, required: true },
  position: { type: String, required: true },
  salary: { type: Number, required: true },
  startDate: { type: Date, required: true },
  manager: { type: mongoose.Schema.Types.ObjectId, ref: 'Employee' },
  skills: [String],
  performanceScore: { type: Number, default: 0 },
  retentionRisk: {
    score: { type: Number, default: 0 },
    level: { type: String, enum: ['low', 'medium', 'high'], default: 'low' },
    factors: [String],
    lastAnalyzed: Date
  },
  sentimentData: {
    currentScore: { type: Number, default: 0 },
    trend: { type: String, enum: ['positive', 'neutral', 'negative'], default: 'neutral' },
    lastUpdated: Date
  },
  ptoBalance: { type: Number, default: 15 },
  ptoUsed: { type: Number, default: 0 },
  status: { type: String, enum: ['active', 'inactive', 'terminated'], default: 'active' }
}, { timestamps: true });

// ============================================
// MODELS/Candidate.js
// ============================================

const candidateSchema = new mongoose.Schema({
  firstName: String,
  lastName: String,
  email: String,
  phone: String,
  position: String,
  resumeText: String,
  resumeUrl: String,
  skills: [String],
  experience: Number,
  education: String,
  aiScore: {
    overall: { type: Number, default: 0 },
    technicalMatch: Number,
    experienceMatch: Number,
    culturalFit: Number,
    details: String
  },
  status: { 
    type: String, 
    enum: ['new', 'screening', 'interview', 'offer', 'rejected', 'hired'],
    default: 'new'
  },
  interviewScheduled: Date,
  notes: [String]
}, { timestamps: true });

// ============================================
// MODELS/PerformanceReview.js
// ============================================

const performanceSchema = new mongoose.Schema({
  employee: { type: mongoose.Schema.Types.ObjectId, ref: 'Employee', required: true },
  reviewPeriod: { type: String, required: true },
  metrics: {
    productivity: Number,
    quality: Number,
    teamwork: Number,
    communication: Number,
    leadership: Number
  },
  overallScore: Number,
  goalsCompleted: Number,
  goalSet: Number,
  strengths: [String],
  areasForImprovement: [String],
  aiRecommendations: [String],
  reviewedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'Employee' },
  reviewDate: { type: Date, default: Date.now }
}, { timestamps: true });

// ============================================
// SERVICES/aiService.js - AI Integration
// ============================================

const OpenAI = require('openai');
const natural = require('natural');

class AIService {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
    this.tokenizer = new natural.WordTokenizer();
    this.TfIdf = natural.TfIdf;
  }

  // Resume Screening AI
  async analyzeResume(resumeText, jobDescription) {
    try {
      const prompt = `
        Analyze this resume against the job description and provide a match score.
        
        Job Description:
        ${jobDescription}
        
        Resume:
        ${resumeText}
        
        Provide analysis in JSON format:
        {
          "overallScore": 0-100,
          "technicalMatch": 0-100,
          "experienceMatch": 0-100,
          "culturalFit": 0-100,
          "strengths": ["strength1", "strength2"],
          "concerns": ["concern1", "concern2"],
          "recommendation": "hire/interview/reject"
        }
      `;

      const response = await this.openai.chat.completions.create({
        model: "gpt-4",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.3
      });

      return JSON.parse(response.choices[0].message.content);
    } catch (error) {
      console.error('AI Resume Analysis Error:', error);
      return this.fallbackResumeAnalysis(resumeText, jobDescription);
    }
  }

  // Fallback NLP-based resume analysis
  fallbackResumeAnalysis(resumeText, jobDescription) {
    const tfidf = new this.TfIdf();
    tfidf.addDocument(jobDescription);
    tfidf.addDocument(resumeText);

    const resumeTokens = this.tokenizer.tokenize(resumeText.toLowerCase());
    const jobTokens = this.tokenizer.tokenize(jobDescription.toLowerCase());
    
    const commonSkills = resumeTokens.filter(token => jobTokens.includes(token));
    const matchPercentage = (commonSkills.length / jobTokens.length) * 100;

    return {
      overallScore: Math.min(matchPercentage, 100),
      technicalMatch: matchPercentage,
      experienceMatch: 70,
      culturalFit: 75,
      strengths: commonSkills.slice(0, 5),
      concerns: [],
      recommendation: matchPercentage > 70 ? 'interview' : 'review'
    };
  }

  // Chatbot Response Generation
  async generateChatbotResponse(userMessage, context = {}) {
    const prompt = `
      You are an AI HR assistant. Answer this employee question professionally and helpfully.
      
      Employee Question: ${userMessage}
      
      Context: ${JSON.stringify(context)}
      
      Provide a clear, concise response. If you need to reference company policies, mention that.
    `;

    try {
      const response = await this.openai.chat.completions.create({
        model: "gpt-4",
        messages: [{ role: "user", content: prompt }],
        max_tokens: 300,
        temperature: 0.7
      });

      return response.choices[0].message.content;
    } catch (error) {
      return this.fallbackChatbotResponse(userMessage);
    }
  }

  fallbackChatbotResponse(message) {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('pto') || lowerMessage.includes('vacation')) {
      return 'You can check your PTO balance in the employee portal. For time off requests, please submit through the system or speak with your manager.';
    } else if (lowerMessage.includes('benefit')) {
      return 'Our benefits include health insurance, 401k, PTO, and more. Visit the benefits portal for detailed information or contact HR.';
    } else {
      return 'I can help with PTO, benefits, policies, and general HR questions. For specific issues, please contact HR directly at hr@company.com.';
    }
  }

  // Sentiment Analysis
  async analyzeSentiment(text) {
    const Analyzer = natural.SentimentAnalyzer;
    const stemmer = natural.PorterStemmer;
    const analyzer = new Analyzer("English", stemmer, "afinn");

    const tokens = this.tokenizer.tokenize(text);
    const score = analyzer.getSentiment(tokens);

    // Score ranges from -5 to 5
    // Convert to 0-100 scale
    const normalizedScore = ((score + 5) / 10) * 100;

    return {
      score: normalizedScore,
      sentiment: score > 1 ? 'positive' : score < -1 ? 'negative' : 'neutral',
      rawScore: score
    };
  }

  // Performance Prediction
  async predictPerformance(employeeData) {
    // Simplified ML model - in production, use real ML
    const {
      goalsCompleted,
      goalsSet,
      attendanceRate,
      peerRating,
      projectsCompleted
    } = employeeData;

    const goalCompletionRate = (goalsCompleted / goalsSet) * 100;
    const performanceScore = (
      goalCompletionRate * 0.3 +
      attendanceRate * 0.2 +
      peerRating * 10 * 0.3 +
      (projectsCompleted / 10) * 100 * 0.2
    );

    return {
      predictedScore: Math.min(performanceScore, 100),
      trend: performanceScore > 80 ? 'improving' : performanceScore < 60 ? 'declining' : 'stable',
      recommendations: this.generatePerformanceRecommendations(performanceScore)
    };
  }

  generatePerformanceRecommendations(score) {
    if (score > 85) {
      return ['Consider for promotion', 'Assign leadership role', 'Mentor junior employees'];
    } else if (score < 65) {
      return ['Schedule 1-on-1 meeting', 'Create performance improvement plan', 'Assign mentor'];
    } else {
      return ['Continue monitoring', 'Provide skill development opportunities', 'Set clear goals'];
    }
  }
}

module.exports = new AIService();

// ============================================
// SERVICES/retentionService.js
// ============================================

class RetentionService {
  async calculateRetentionRisk(employee) {
    let riskScore = 0;
    const factors = [];

    // Salary below market (weight: 30)
    if (employee.salaryBelowMarket > 10) {
      riskScore += 30;
      factors.push('Salary below market rate');
    }

    // No promotion in 2+ years (weight: 20)
    const yearsSincePromotion = this.getYearsSincePromotion(employee);
    if (yearsSincePromotion > 2) {
      riskScore += 20;
      factors.push('No recent promotion');
    }

    // Low engagement (weight: 25)
    if (employee.engagementScore < 6) {
      riskScore += 25;
      factors.push('Low engagement score');
    }

    // High workload (weight: 15)
    if (employee.weeklyHours > 50) {
      riskScore += 15;
      factors.push('High workload');
    }

    // Team turnover (weight: 10)
    if (employee.teamTurnoverRate > 20) {
      riskScore += 10;
      factors.push('High team turnover');
    }

    const level = riskScore > 70 ? 'high' : riskScore > 40 ? 'medium' : 'low';

    return {
      score: riskScore,
      level,
      factors,
      recommendations: this.generateRecommendations(riskScore, factors)
    };
  }

  generateRecommendations(score, factors) {
    const recommendations = [];

    if (factors.includes('Salary below market rate')) {
      recommendations.push('Review and adjust compensation');
    }
    if (factors.includes('No recent promotion')) {
      recommendations.push('Discuss career development path');
    }
    if (factors.includes('Low engagement score')) {
      recommendations.push('Schedule 1-on-1 to understand concerns');
    }
    if (factors.includes('High workload')) {
      recommendations.push('Review workload distribution');
    }

    return recommendations;
  }

  getYearsSincePromotion(employee) {
    // Placeholder - implement based on employee history
    return 2;
  }

  async analyzeBatch() {
    const Employee = mongoose.model('Employee');
    const employees = await Employee.find({ status: 'active' });

    for (const employee of employees) {
      const risk = await this.calculateRetentionRisk(employee);
      employee.retentionRisk = {
        ...risk,
        lastAnalyzed: new Date()
      };
      await employee.save();
    }

    console.log(`✅ Analyzed retention risk for ${employees.length} employees`);
  }
}

module.exports = new RetentionService();

// ============================================
// ROUTES/recruitment.js
// ============================================

const router = require('express').Router();
const multer = require('multer');
const pdfParse = require('pdf-parse');
const AIService = require('../services/aiService');

const upload = multer({ storage: multer.memoryStorage() });

// Upload and analyze resume
router.post('/upload-resume', upload.single('resume'), async (req, res) => {
  try {
    const { position, jobDescription } = req.body;
    const pdfBuffer = req.file.buffer;

    // Extract text from PDF
    const pdfData = await pdfParse(pdfBuffer);
    const resumeText = pdfData.text;

    // AI Analysis
    const analysis = await AIService.analyzeResume(resumeText, jobDescription);

    // Extract basic info (simplified)
    const emailMatch = resumeText.match(/[\w.-]+@[\w.-]+\.\w+/);
    const phoneMatch = resumeText.match(/\d{3}[-.]?\d{3}[-.]?\d{4}/);

    const Candidate = mongoose.model('Candidate');
    const candidate = new Candidate({
      email: emailMatch ? emailMatch[0] : '',
      phone: phoneMatch ? phoneMatch[0] : '',
      position,
      resumeText,
      skills: analysis.strengths,
      aiScore: {
        overall: analysis.overallScore,
        technicalMatch: analysis.technicalMatch,
        experienceMatch: analysis.experienceMatch,
        culturalFit: analysis.culturalFit,
        details: JSON.stringify(analysis)
      }
    });

    await candidate.save();

    res.json({
      success: true,
      candidate: candidate._id,
      analysis
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all candidates
router.get('/candidates', async (req, res) => {
  try {
    const Candidate = mongoose.model('Candidate');
    const candidates = await Candidate.find()
      .sort({ 'aiScore.overall': -1 })
      .limit(50);

    res.json(candidates);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Schedule interview
router.post('/schedule-interview/:id', async (req, res) => {
  try {
    const Candidate = mongoose.model('Candidate');
    const candidate = await Candidate.findById(req.params.id);

    candidate.interviewScheduled = req.body.date;
    candidate.status = 'interview';
    await candidate.save();

    // Send email notification (implement with nodemailer)
    
    res.json({ success: true, candidate });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;

// ============================================
// ROUTES/chatbot.js
// ============================================

router.post('/message', async (req, res) => {
  try {
    const { message, employeeId } = req.body;

    const Employee = mongoose.model('Employee');
    const employee = await Employee.findById(employeeId);

    const context = {
      ptoBalance: employee?.ptoBalance,
      department: employee?.department,
      position: employee?.position
    };

    const response = await AIService.generateChatbotResponse(message, context);

    res.json({
      message: response,
      timestamp: new Date()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;

// ============================================
// ROUTES/performance.js
// ============================================

router.get('/employee/:id', async (req, res) => {
  try {
    const Performance = mongoose.model('PerformanceReview');
    const reviews = await Performance.find({ employee: req.params.id })
      .sort({ reviewDate: -1 })
      .limit(5);

    res.json(reviews);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/predict/:id', async (req, res) => {
  try {
    const Employee = mongoose.model('Employee');
    const employee = await Employee.findById(req.params.id);

    const prediction = await AIService.predictPerformance({
      goalsCompleted: req.body.goalsCompleted,
      goalsSet: req.body.goalsSet,
      attendanceRate: req.body.attendanceRate || 95,
      peerRating: req.body.peerRating || 8,
      projectsCompleted: req.body.projectsCompleted || 5
    });

    res.json(prediction);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;

// ============================================
// .env.example - Environment Variables
// ============================================

/*
PORT=5000
MONGODB_URI=mongodb://localhost:27017/ai-hr-manager
JWT_SECRET=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-email-password
*/
