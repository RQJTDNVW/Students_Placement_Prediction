import { z } from 'zod'

export const studentSchema = z.object({
  cgpa: z.number({ error: 'Enter a CGPA between 0 and 10.' }).min(0, 'CGPA cannot be below 0.').max(10, 'CGPA cannot exceed 10.'),
  aptitude_score: z.number({ error: 'Enter an aptitude score between 0 and 100.' }).min(0, 'Score cannot be below 0.').max(100, 'Score cannot exceed 100.'),
  internships: z.number({ error: 'Enter a whole number of internships.' }).int('Use a whole number.').min(0, 'Internships cannot be negative.'),
  projects_count: z.number({ error: 'Enter a whole number of projects.' }).int('Use a whole number.').min(0, 'Projects cannot be negative.'),
  certifications: z.number({ error: 'Enter a whole number of certifications.' }).int('Use a whole number.').min(0, 'Certifications cannot be negative.'),
  communication_skills: z.number({ error: 'Enter a communication score between 0 and 10.' }).min(0, 'Score cannot be below 0.').max(10, 'Score cannot exceed 10.'),
  extracurricular_activities: z.number({ error: 'Enter an extracurricular score between 0 and 10.' }).min(0, 'Score cannot be below 0.').max(10, 'Score cannot exceed 10.'),

  // Optional technical fields for V4 model
  coding_skills: z.number().min(0).max(10).optional(),
  dsa_score: z.number().min(0).max(10).optional(),
  backlogs: z.number().int().min(0).optional(),
  college_tier: z.enum(['Tier-1', 'Tier-2', 'Tier-3']).optional(),
  branch: z.string().optional(),
  hackathons: z.number().int().min(0).optional(),
  open_source_contributions: z.number().int().min(0).optional(),
  system_design: z.number().min(0).max(10).optional(),
  ml_knowledge: z.number().min(0).max(10).optional(),
})

